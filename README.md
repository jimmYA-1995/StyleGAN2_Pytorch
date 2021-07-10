# StyleGAN2_Pytorch
reimplementation of stylegan2 with pytorch
The original repository from: https://github.com/NVlabs/stylegan2


## Train
example:
``` bash
# single GPU
OMP_NUM_THREADS=1 python run_training.py --cfg config.yml

# multi GPU training (example 4)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_training.py --cfg cfg.yml

# enable AMP & gradient scaling to accelerate training
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_training.py --cfg cfg.yml --autocast --gradscale

# provide env. for resuming logging on weight & bias
WANDB_RESUME=must WANDB_RUN_ID=<wandb-run-id> python -m torch.distributed.launch --nproc_per_node=4 run_training.py --cfg cfg.yml
``` 

note

* set `OMP_NUM_THREADS=1` to prevent scipy performance issue in multi-processing. (automatically set by pytorch distributed launch script)
* `wandb` is enable by default, add `--no-wandb` flag to disable. I also provide output log and FID metirc in local run directory.
* format of run directory: `<serial-number>-<wandb-run-id>-<num-gpus>-<config-filename>`
* when resuming from checkpoint, starting iteration will auto set by checkpoint filename. (`ckpt-187500.pt` means 187500 iterations.)
