echo $MLP_WORKER_0_HOST
echo $MLP_ROLE_INDEX
echo $MLP_WORKER_0_PORT
echo $TORCH_USE_CUDA_DSA


accelerate launch --config_file distribute_configs/pixart_t2v_huoshan.yaml \
    --main_process_ip $MLP_WORKER_0_HOST \
    --machine_rank $MLP_ROLE_INDEX \
    --main_process_port $MLP_WORKER_0_PORT pipeline/extract_feature/extract_video_feature.py \
    --num_frames 180 \
    --max_num_sample 300000

# CUDA_VISIBLE_DEVICES='4,5,6,7' accelerate launch --config_file distribute_configs/pixart_t2v_huoshan.yaml  pipeline/extract_feature/extract_video_feature.py
