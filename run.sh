function get_random_port() {
    port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    echo $port
}


gpus=$1
choice=$2
shift;shift # remove first two arguments

echo "Provided arguments:"
echo "gpus: $gpus"
echo "choices: $choice"
echo "-------"

# if choice contains "mask", start masked ddpm training
if [[ "$choice" == *"mask"*  ]];then

    params=$1
    params=(${params//,/ });shift # seperate argument by comma
    name=${params[0]}
    config=${params[1]}

    echo "name of Mask training: $name, config file: $config"
    echo "Start masked training" 

    CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --mixed_precision fp16 --num_cpu_threads_per_process 5 \
            --num_machines 1 --gpu_ids $gpus --multi_gpu  \
            main.py \
            --config $config \
            --name $name \
            --debug \
            --train_steps 200000 \
            --wandb_project diffusion

fi
# if choice contains "ddpm", start ddpm training
if [[ "$choice" == *"ddpm"*  ]];then

    # echo "$1"
    # ramdom_port="$(get_random_port)"
    params=$1
    params=(${params//,/ });shift # seperate argument by comma

    name=${params[0]}
    config=${params[1]}
    pretrained_model_ckpt=${params[2]}

    echo "Start DDPM training" 
    echo "name of DDPM training: $name"
    echo "config file: $config"
    echo "pretrained weight: $pretrained_model_ckpt"

    CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
        --mixed_precision fp16 --num_cpu_threads_per_process 10\
        --num_machines 1 --gpu_ids $gpus --multi_gpu \
        main.py \
        --pretrained_model_ckpt "$pretrained_model_ckpt" \
        --config $config \
        --name $name \
        --debug \
        --train_steps 200000 \
        --wandb_project diffusion
fi
# if choice contains "test", start sampling
if [[ "$choice" == *"test"*  ]];then

    ckpt_lst=(1 5 10 15 20)
    output_dir="/data/shared/output/jiachen/mask-uvit"
    post_fix=""
    params=$1
    params=(${params//,/ });shift # seperate argument by comma

    exp_name=${params[0]}
    config=${params[1]}

    echo "Start testing: $exp_name" 

    ckpt_lst_str=""
    output_lst_str=""
    for ckpt in ${ckpt_lst[@]}
    do
        mkdir -p $output_dir/$exp_name/eval/"$ckpt"0k/$post_fix
        ckpt_lst_str+=" $output_dir/$exp_name/model-$ckpt.pt"
        output_lst_str+=" "
        output_lst_str+=$output_dir/$exp_name/eval/"$ckpt"0k/$post_fix
    done

    CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
        --mixed_precision fp16 --num_cpu_threads_per_process 10 \
        --num_machines 1 --gpu_ids $gpus --multi_gpu\
        eval.py \
        --bs 64 \
        --sampling_steps 250 \
        --ddim_sampling_eta 1.0 \
        --total_samples 10000 \
        --config $config \
        --ckpt $ckpt_lst_str \
        --output $output_lst_str \
        --sampler ddim
fi