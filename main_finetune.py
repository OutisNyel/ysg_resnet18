from util.argument import get_args_parser
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup

from resnet18_ysg import DualResNet
import util.misc as misc
from datasets_ysg import FundusDataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from loss_ysg import AsymmetricLossWithWeight
from engine_finetune import train_one_epoch, evaluate

import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args, criterion=AsymmetricLossWithWeight(), checkpoint_path=''):
    if args.resume and not args.eval:
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        args = checkpoint['args']
        args.resume = resume

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dual_res = DualResNet()

    # 调整模型
    # if not args.eval: #tmp
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)

    checkpoint_model = checkpoint.get('model', checkpoint)

    # load pre-trained model
    msg = dual_res.load_state_dict(checkpoint_model, strict=False)

    dataset_train_and_val = FundusDataset(
        is_train='train',
        args=args,
    )
    dataset_test = FundusDataset(
        is_train='test',
        args=args,
        sql='''
SELECT
    left_path,
    right_path,
    diabetic,
    glaucoma,
    cataract,
    age_related_macular_degeneration,
    hypertensive_retinopathy,
    myopia,
    other_diseases
FROM 
    fundus
WHERE 
    is_test = 1;
'''
    )

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    elif args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task+'_test'))
    else:
        log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_labels)

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Load checkpoint from: %s" % args.resume)
        dual_res.load_state_dict(checkpoint['model'])

    dual_res.to(device)
    model_without_ddp = dual_res

    n_parameters = sum(p.numel() for p in dual_res.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.AdamW(dual_res.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    print("criterion = %s" % str(criterion))

    # ddp stands for distributed data parallel
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        (test_stats, test_score) = evaluate(
            data_loader_test,
            dual_res,
            criterion,
            device,
            args,
            epoch=0,
            mode='test',
            log_writer=log_writer
        )
        exit(0)

    '''Start training'''
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_epoch = 0
    max_score = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_set, val_set = torch.utils.data.random_split(dataset_train_and_val, [0.8, 0.2])
        data_loader_train = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # 训练一轮
        train_stats = train_one_epoch(
            dual_res,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args
        )
        # 验证一次
        (val_stats, val_score) = evaluate(
            data_loader_val,
            dual_res,
            criterion,
            device,
            args,
            epoch,
            mode='val',
            log_writer=log_writer
        )
        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args,
                    model=dual_res,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    mode='best'
                )
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))
        # 保存最新模型
        if args.output_dir and args.savemodel:
            misc.save_model(
                args=args,
                model=dual_res,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                mode='latest'
            )
        print("Save the latest model.")

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.task, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # 最后一轮评估最佳模型
        if epoch == (args.epochs - 1):
            checkpoint = torch.load(os.path.join(args.output_dir, args.task, 'checkpoint-best.pth'), map_location='cpu',
                                    weights_only=False)
            dual_res.load_state_dict(checkpoint['model'], strict=False)
            dual_res.to(device)
            print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
            (test_stats, test_score) = evaluate(
                data_loader_test,
                dual_res, criterion,
                device,
                args,
                -1,
                mode='test',
                log_writer=None
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    criterion = AsymmetricLossWithWeight().to(args.device)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = r'C:\Users\Outis/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth'
    main(args, criterion, CHECKPOINT_PATH)


