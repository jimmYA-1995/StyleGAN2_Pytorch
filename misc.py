import os
import time
import shutil
import logging
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """ Logging Formatter  to add colors and count warning / errors"""
    def __init__(self, name):
        gray = "\x1b[38;21m"
        green = "\x1b[32;21m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = f"<{name}> %(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"

        self.FORMATS = {
            'DEBUG': gray + format + reset,
            'INFO': green + format + reset,
            'WARNING': yellow + format + reset,
            'ERROR': red + format + reset,
            'CRITICAL': bold_red + format + reset,
        }
        super(CustomFormatter, self).__init__()

    def format(self, record):
        log_fmt = self.FORMATS[record.levelname]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def create_logger(out_dir, rank, debug=False):
    logger_name = None if rank == 0 else f"GPU{rank}"
    ctx_name = logger_name if logger_name is not None else 'GPU0'
    loglevel = 'DEBUG' if debug else ('INFO' if rank == 0 else 'WARN')

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, loglevel))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, loglevel))
    ch.setFormatter(CustomFormatter(ctx_name))
    logger.addHandler(ch)
    
    # disable PIL dubug mode
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    
    if rank == 0:
        # TODO: logging to a single file from multiple processes
        # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        filename = out_dir / 'experiment.log'
        head = "%(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
        fh = logging.FileHandler(filename)
        fh.setLevel(getattr(logging, loglevel))
        fh.setFormatter(head)
        logger.addHandler(fh)

    return logger


# def validate_configuration(args, cfg):
#     assert self.num_classes >= 1


def prepare_training(args, cfg):
    """ populate necessary directories """
    root_dir = Path(cfg.OUT_DIR)
    if not root_dir.exists():
        print('creating {}'.format(root_dir))
        root_dir.mkdir(parents=True)

    cfg_name = os.path.basename(args.cfg).split('.')[0] if args.cfg else 'default'
    exist_IDs = [int(x.name[:5])
                 for x in root_dir.glob("[0-9][0-9][0-9][0-9][0-9]-*")
                 if x.is_dir()]
    exp_ID = max(exist_IDs) + 1 if exist_IDs else 0
    exp_ID = str(exp_ID).zfill(5)

    out_dir = root_dir / f'{exp_ID}-{args.num_gpus}gpu-{cfg_name}'
    (out_dir / 'checkpoints').mkdir(parents=True)
    (out_dir / 'samples').mkdir(parents=True)
    args.out_dir = out_dir
