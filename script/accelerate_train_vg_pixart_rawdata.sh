echo $MLP_WORKER_0_HOST
echo $MLP_ROLE_INDEX
echo $MLP_WORKER_0_PORT
echo $TORCH_USE_CUDA_DSA

export GPUS_PER_NODE=${MLP_WORKER_GPU}
export NNODES=${MLP_WORKER_NUM}
export NODE_RANK=${MLP_ROLE_INDEX}
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export MASTER_PORT=${MLP_WORKER_0_PORT}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
export NCCL_DEBUG=WARN

accelerate launch --config_file distribute_configs/pixart_t2v_huoshan.yaml \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --mixed_precision bf16 \
    --main_process_port $MASTER_PORT pipeline/trainer_vg_rawdata.py \
    --project_name 'vg_replacesdxl_488vae_pixar_sigma_mix_frame120_bs4_t4' \
    --video_dataset_name 'pixar_sigma_mix' \
    --num_frame 120 \
    --train_batch_size 4
# CUDA_VISIBLE_DEVICES='4,5,6,7' accelerate launch --config_file distribute_configs/pixart_t2v_huoshan.yaml  pipeline/trainer_vg_rawdata.py 
