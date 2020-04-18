import argparse


class CustomedFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
    pass

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Train Style GANv2',
        # epilog=examples,
        formatter_class=CustomedFormatter)

    parser.add_argument('path', type=str)
    parser.add_argument('--iter', type=int, default=800000, help='training iteration')
    parser.add_argument('--batch', type=int, default=16, help='mini-batch size')
    parser.add_argument('--n_mlp', type=int, default=8, help='#layer of mapping network')
    parser.add_argument('--latent', type=int, default=512, help='latent dimension')
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2) ##
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--result-dir', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--skeleton_channel', type=int, default=0, help='the num of channel used for skeleton')

    return parser.parse_args()
