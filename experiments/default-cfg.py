description = "Default configuration file"
config = {
    'gpus': [0,1,2,3,4,5,6,7],
    'resolution': 256,
    'dataset': {
        'root': [],
        'n_worker': 2,
        'batch_size': 16,
    },
    'model': {
        'n_mlp': 8,
        'latent': 512,
        'channel_multiplier': 2,
        'extra_channels': 2,
    },
    'train': {
        'iter': 120000,
        'lr': 1e-3,
        'r1': 10,
        'path_regularize': 2,
        'path_batch_shrink': 2,
        'g_reg_every': 4,
        'd_reg_every': 16,
        'style_mixing': 0.9,
        'ckpt': None,
    },
    'n_sample': 64,
    'output_dir': 'results',
}
