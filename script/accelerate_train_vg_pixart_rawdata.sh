echo $MLP_WORKER_0_HOST
echo $MLP_ROLE_INDEX
echo $MLP_WORKER_0_PORT
echo $TORCH_USE_CUDA_DSA


accelerate launch --config_file distribute_configs/pixart_t2v_huoshan.yaml \
    --main_process_ip $MLP_WORKER_0_HOST \
    --machine_rank $MLP_ROLE_INDEX \
    --main_process_port $MLP_WORKER_0_PORT pipeline/trainer_vg_rawdata.py \
    --project_name 'vg_replacesdxl_488vae_i+v_t1'


