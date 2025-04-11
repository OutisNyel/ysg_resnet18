import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE fine-tuning for image classification',
        add_help=False)
    parser.add_argument(
        '--batch_size', default=8, type=int,
        help='Batch size per GPU '
        '(effective batch size is batch_size * accum_iter * # gpus)'
        '16 suits for RTX 4060.')
    parser.add_argument(
        '--epochs',default=50, type=int,
        help='Official recommends 50 epochs')
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations '
             '(for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument(
        '--model', default='vit_large_patch16', type=str,
        help='Name of model to train',
        metavar='MODEL') # metavar 占位符
    parser.add_argument(
        '--input_size', default=224, type=int,
        help='images input size')
    parser.add_argument(
        '--drop_path', type=float, default=0.2, metavar='PCT',
        help='Drop path rate (default: 0.1). 随机丢弃神经网络路径率')

    # Optimizer parameters
    parser.add_argument(
        '--clip_grad', type=float, default=None,
        help='Clip gradient norm (default: None, no clipping)',
        metavar='NORM')
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='weight decay (default: 0.05). 权重损失'
             'loss = loss0 + 0.5 * weight_decay * weight^2')
    parser.add_argument(
        '--lr', type=float, default=None,
        help='learning rate (absolute lr)',
        metavar='LR')
    parser.add_argument(
        '--blr', type=float, default=5e-4,
        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256',
        metavar='LR')
    parser.add_argument(
        '--layer_decay', type=float, default=0.65,
        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument(
        '--min_lr', type=float, default=1e-6,
        help='lower lr bound for cyclic schedulers that hit 0. '
             'Cyclic scheduler 循环学习率调度器',
        metavar='LR')
    parser.add_argument(
        '--warmup_epochs', type=int, default=2,
        help='epochs to warmup LR',
        metavar='N')

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter', type=float, default=None,
        help='Color jitter factor (enabled only when not using Auto/RandAugment)',
        metavar='PCT')
    parser.add_argument(
        '--aa', type=str, default='rand-m9-mstd0.5-inc1',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
             'AutoAugment, RandAugment 是两种主流的自动化数据增强策略',
        metavar='NAME')
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing (default: 0.1)'
             '标签平滑，防止过拟合。[1,0,0]->[.9,.05,.05]'
             'smoothed_label = (1-ε) * one_hot_label + ε/(K-1) * (1-one_hot_label)')

    # * Random Erase params
    parser.add_argument(
        '--reprob', type=float, default=0.25,
        help='Random erase probability (default: 0.25)',
        metavar='PCT')
    parser.add_argument(
        '--remode', type=str, default='pixel',
        help='Random erase mode (default: "pixel")')
    parser.add_argument(
        '--recount', type=int, default=1,
        help='Random erase count (default: 1)')
    parser.add_argument(
        '--resplit', action='store_true', default=False, # action = store_true 意味着出现 --resplit 该项为真
        help='Do not random erase first (clean) augmentation split'
             '不在第一个（未经处理的）数据增强分割中执行随机擦除')

    # * Mixup params 混合增强, 默认不开启
    parser.add_argument(
        '--mixup', type=float, default=0,
        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument(
        '--cutmix', type=float, default=0,
        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument(
        '--cutmix_minmax', type=float, nargs='+', default=None,
        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument(
        '--mixup_prob', type=float, default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', type=str,
        help='finetune from checkpoint')
    parser.add_argument(
        '--task', default='', type=str,
        help='finetune from checkpoint')
    parser.add_argument(
        '--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        '--cls_token', action='store_false', dest='global_pool',
        help='Use class token instead of global pool for classification')

    # Dataset parameters ARCHIVED mostly
    parser.add_argument( # ARCHIVED. Load from database
        '--data_path', default='./data/', type=str,
        help='dataset path')
    parser.add_argument( # ARCHIVED. Multi-label task
        '--nb_labels', default=7, type=int,
        help='number of the classification types')
    parser.add_argument(
        '--output_dir', default='./output_dir',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default='./output_logs',
        help='path where to tensorboard log')
    parser.add_argument(
        '--device', default='cuda',
        help='device to use for training / testing')
    parser.add_argument(
        '--seed', default=0, type=int)
    parser.add_argument(
        '--resume', default='',
        help='resume from checkpoint')
    parser.add_argument(
        '--start_epoch', default=0, type=int,
        help='start epoch',
        metavar='N')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform evaluation only')
    # dist below stands for distributed
    parser.add_argument(
        '--dist_eval', action='store_true', default=False,
        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument(
        '--num_workers', default=2, type=int)
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size', default=1, type=int,
        help='number of distributed processes')
    parser.add_argument(
        '--local_rank', default=-1, type=int)
    parser.add_argument(
        '--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url', default='env://',
        help='url used to set up distributed training')

    # fine-tuning parameters
    parser.add_argument(
        '--savemodel', action='store_true', default=True,
        help='Save model')
    parser.add_argument(
        '--norm', default='IMAGENET', type=str,
        help='Normalization method')
    parser.add_argument(
        '--enhance', action='store_true', default=False,
        help='Use enhanced data')
    parser.add_argument(
        '--datasets_seed', default=2026, type=int)

    # 预处理
    args = parser.parse_args()

    if args.warmup_epochs > args.epochs // 5:
        args.warmup_epochs = args.epochs // 5
    if args.warmup_epochs > 10:
        args.warmup_epochs = 10

    return parser