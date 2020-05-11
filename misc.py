import os
import time
import logging
import argparse
from pathlib import Path

from pprint import pprint


def parse_args(arg=None):
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--debug", action='store_true', default=False, help="whether to use debug mode")
    parser.add_argument("--cfg", default='experiments/default-cfg.py', help="path to the configuration file")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    
    if arg:
        args, _ = parser.parse_known_args(arg.split())
    else:
        args, _ = parser.parse_known_args()

    return args

def create_logger(out_dir):
    final_log_file = out_dir / 'experiment.log'
    head = '%(levelname)-8s %(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def prepare_training(cfg, cfg_path):
    """
    ?? how to control logger in multi-processing
    """
    root_out_dir = Path(cfg.OUT_DIR)
    if not root_out_dir.exists():
        print('creating {}'.format(root_out_dir))
        root_out_dir.mkdir(parents=True)
    
    # create run directory
    run_id = '00000'
    cfg_name = os.path.basename(cfg_path).split('.')[0]
    ids = [int(str(x).split('/')[-1][:5]) for x in root_out_dir.glob("[0-9][0-9][0-9][0-9][0-9]-*")]
    if len(ids) > 0:
        run_id = str(sorted(ids)[-1] + 1).zfill(5)
    
    final_out_dir = root_out_dir / f'{run_id}-{len(cfg.GPUS)}gpu-{cfg_name}'
    final_out_dir.mkdir()
    
    # out of expectation
    with open(final_out_dir / 'configuration.txt', 'w') as f:
        pprint(cfg, f)
    
    (final_out_dir / 'checkpoints').mkdir()
    (final_out_dir / 'samples').mkdir()
    
    logger = create_logger(final_out_dir)
    
    return logger, final_out_dir