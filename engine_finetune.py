import numpy as np
import os
import torch
from typing import Iterable, Optional, Dict, Tuple
from timm.data import Mixup
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import util.misc as misc
import util.lr_sched as lr_sched
from util.safe_auc import safe_auc
from util.get_confusion_matrix import get_confusion_matrix

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
) -> Dict:
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = (samples[0].to(device, non_blocking=True), samples[1].to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        
        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(
        data_loader,
        model,
        criterion,
        device,
        args,
        epoch,
        mode,
        log_writer
) -> Tuple[Dict, float]:
    """Evaluate the model."""
    CLS = [ 'D', 'G', 'C', 'A', 'H', 'M', 'O' ]
    # 设置分割符
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()

    data_len = len(data_loader.dataset)
    targets = np.zeros((data_len, len(CLS)))
    predicts = np.zeros((data_len, len(CLS)))
    predicts_binary = np.zeros((data_len, len(CLS)))
    index = 0
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images = (batch[0][0].to(device, non_blocking=True),batch[0][0].to(device, non_blocking=True))
        target = batch[-1].to(device, non_blocking=True)
        batch_size = target.shape[0]
        
        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)
        output_ = torch.sigmoid(output)
        output_binary = (output_ > 0.5).float()

        metric_logger.update(loss=loss.item())
        targets[index, index + batch_size:]         = target.cpu().numpy()
        predicts[index, index + batch_size:]        = output_.detach().cpu().numpy()
        predicts_binary[index, index + batch_size:] = output_binary.detach().cpu().numpy()
        index += batch_size

    tps = np.zeros(len(CLS))
    fns = np.zeros(len(CLS))
    fps = np.zeros(len(CLS))
    tns = np.zeros(len(CLS))
    accuracies = np.zeros(len(CLS))
    precisions = np.zeros(len(CLS))
    recalls = np.zeros(len(CLS))
    f1s = np.zeros(len(CLS))
    aucs = np.zeros(len(CLS))
    probs = np.zeros(len(CLS))
    variances = np.zeros(len(CLS))
    for class_index in range(len(CLS)):
        target_c = targets[:, class_index]
        predicts_b_c = predicts_binary[:, class_index]
        predicts_c = predicts[:, class_index]

        tp, fn, fp, tn = get_confusion_matrix(target_c, predicts_b_c)
        tps[class_index] = tp
        fns[class_index] = fn
        fps[class_index] = fp
        tns[class_index] = tn
        accuracies[class_index] = accuracy_score(target_c, predicts_b_c)
        precisions[class_index] = precision_score(target_c, predicts_b_c, zero_division=0)
        recalls[class_index] = recall_score(target_c, predicts_b_c, zero_division=0)
        f1s[class_index] = f1_score(target_c, predicts_b_c, zero_division=0)
        aucs[class_index] = safe_auc(target_c, predicts_c)
        probs[class_index] = predicts_c.mean()
        variances[class_index] = np.var(predicts_c)

    f1 = f1s.mean()
    roc_auc = aucs.mean()

    score = (f1 + roc_auc) / 2

    if log_writer:
        for i, cls in enumerate(CLS):
            log_writer.add_scalar(f'{cls}/TP', tps[i], epoch)
            log_writer.add_scalar(f'{cls}/FN', fns[i], epoch)
            log_writer.add_scalar(f'{cls}/FP', fps[i], epoch)
            log_writer.add_scalar(f'{cls}/TN', tns[i], epoch)
            log_writer.add_scalar(f'{cls}/Accuracy', accuracies[i], epoch)
            log_writer.add_scalar(f'{cls}/Precision', precisions[i], epoch)
            log_writer.add_scalar(f'{cls}/Recall', recalls[i], epoch)
            log_writer.add_scalar(f'{cls}/Probability', probs[i], epoch)
            log_writer.add_scalar(f'{cls}/ROC_AUC', aucs[i], epoch)
            log_writer.add_scalar(f'{cls}/Variance', variances[i], epoch)
        for metric_name, value in zip(['f1','roc_auc', 'score'],
                                       [f1, roc_auc, score]):
            log_writer.add_scalar(f'Performance/{metric_name}', value, epoch)

    # print
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Score: {score:.4f}')
    print('Class |Accura|Precis|Recall|TP    |FN    |FP    |TN    |AUC   |Varian|Probab')
    for i, cls in enumerate(CLS):
        print(f'{cls}     |{accuracies[i]:.4f}|{precisions[i]:.4f}|{recalls[i]:.4f}|{tps[i]:.5d} |{fns[i]:.5d} |{fps[i]:.5d} |{tns:.5d}|{aucs[i]:.4f}|{variances[i]:.4f}|{probs[i]:.4f}|{probs[i]:.4f}')
    
    metric_logger.synchronize_between_processes()
    
    return ({k: meter.global_avg
             for k, meter in metric_logger.meters.items()},
            score)
