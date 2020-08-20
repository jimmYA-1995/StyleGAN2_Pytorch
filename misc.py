import os
import time
import logging
import argparse
from pathlib import Path

from pprint import pprint


class CustomFormatter(logging.Formatter):
    """ Logging Formatter  to add colors and count warning / errors"""
    
    gray = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "<MASTER> %(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
    
    FORMATS = {
        'DEBUG': gray + format + reset,
        'INFO': green + format + reset,
        'WARNING': yellow + format + reset,
        'ERROR': red + format + reset,
        'CRITICAL': bold_red + format + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS[record.levelname]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

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

def create_logger(out_dir, level='INFO'):
    log_file = out_dir / 'experiment.log'
    head = "%(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
    logging.basicConfig(filename=str(log_file),
                       format=head,
                       level=logging.DEBUG)
    logger = logging.getLogger()
    
    print(f"level: {level}")
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level, 'INFO'))
    console.setFormatter(CustomFormatter())
    logger.addHandler(console)

    return logger


def prepare_training(cfg, cfg_path, debug=False):
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
    
    final_out_dir = root_out_dir / f'{run_id}-{len(os.environ["CUDA_VISIBLE_DEVICES"])}gpu-{cfg_name}'
    final_out_dir.mkdir()
    
    # out of expectation
    with open(final_out_dir / 'configuration.txt', 'w') as f:
        pprint(cfg, f)
    
    (final_out_dir / 'checkpoints').mkdir()
    (final_out_dir / 'samples').mkdir()
    
    loglevel = 'DEBUG' if debug else 'INFO'
    logger = create_logger(final_out_dir, loglevel)
    
    return logger, final_out_dir

    
    
    