export OMP_NUM_THREADS=3
export NUM_PROCESS=2
export CUDA_VISIBLE_DEVICES='0,1'
python -m torch.distributed.launch \
	--nproc_per_node=${NUM_PROCESS} \
	run_training.py \
	--cfg experiments/append_to_fm_512.yml \
	--debug \
	#--wandb

