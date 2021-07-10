import os
import logging
from pathlib import Path
from typing import Any


class UserError(Exception):
    pass

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class CustomFormatter(logging.Formatter):
    """ Logging Formatter  to add colors and count warning / errors"""
    def __init__(self, name):
        gray     = "\x1b[38;21m"
        green    = "\x1b[32;21m"
        yellow   = "\x1b[33;21m"
        red      = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset    = "\x1b[0m"
        format   = f"<{name}> %(message)s (%(filename)s:%(lineno)d)"

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


def create_logger(local_rank, out_dir=None, debug=False, **kwargs):
    if out_dir is not None:
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    logger_name = f"GPU{local_rank}"
    loglevel = 'DEBUG' if debug else ('INFO' if local_rank == 0 else 'WARN')

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(getattr(logging, loglevel))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, loglevel))
    ch.setFormatter(CustomFormatter(logger_name))
    logger.addHandler(ch)

    # disable PIL dubug mode
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    if local_rank == 0 and out_dir:
        # TODO: logging to a single file from multiple processes
        # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        assert out_dir is not None
        filename = out_dir / 'experiment.log'
        head = "%(levelname)-8s - %(asctime)-15s - %(message)s (%(filename)s:%(lineno)d)"
        fh = logging.FileHandler(filename)
        fh.setLevel(getattr(logging, loglevel))
        fh.setFormatter(logging.Formatter(head))
        logger.addHandler(fh)

    return logger


def prepare_training(args, cfg):
    """ populate necessary directories """
    if not args.out_dir:
        root_dir = Path(cfg.outdir)
        if not root_dir.exists():
            print('creating {}'.format(root_dir))
            root_dir.mkdir(parents=True)

        cfg_name = cfg.name if cfg.name else 'default'
        existing_serial_num = [int(x.name[:5])
                               for x in root_dir.glob("[0-9][0-9][0-9][0-9][0-9]-*")
                               if x.is_dir()]
        serial_num = max(existing_serial_num) + 1 if existing_serial_num else 0
        serial_num = str(serial_num).zfill(5)

        args.out_dir = root_dir / f'{serial_num}-{args.wandb_id}-{args.num_gpus}gpu-{cfg_name}'

    (args.out_dir / 'checkpoints').mkdir(parents=True)
    (args.out_dir / 'samples').mkdir(parents=True)
