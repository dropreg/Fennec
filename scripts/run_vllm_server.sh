#!/bin/bash

MODEl_NAME=/public/home/ljt/lxb/hf_lfs/models/Mixtral-8x7B-Instruct-v0.1
# MODEl_NAME=/public/home/ljt/hf_models/Llama-2-13b-chat-hf
# MODEl_NAME=/public/home/ljt/hf_models/Llama-2-7b-chat-hf
# MODEl_NAME=/public/home/ljt/lxb/hf_lfs/Mistral-7B-Instruct-v0.2
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64
export RAY_memory_monitor_refresh_ms=0

EVAL_PARALLEL=true

echo "run vllm server parallel..."

cuda_list=(0 1 2 3 4 5 6 7)
port_list=(8000 8001 8002 8003 8004 8005 8006 8007)
parallel=8

if [ "$EVAL_PARALLEL" == "true" ]; then
    length=${#cuda_list[@]}
else
    length=1
fi

for ((i=0; i<$length; i=i+$parallel)); do
    port=${port_list[$i]}
    cuda=""
    for ((j=0; j<$parallel; j++))
    do
        if [ $j -eq 0 ]; then
            cuda="${cuda_list[$i+j]}"
        else
            cuda="$cuda,${cuda_list[$i+j]}"
        fi
    done

    # CUDA_VISIBLE_DEVICES=$cuda python -m vllm.entrypoints.openai.api_server \
    CUDA_VISIBLE_DEVICES=$cuda python -m vllm.entrypoints.api_server \
    --port $port \
    --model $MODEl_NAME \
    --tokenizer $MODEl_NAME \
    --tensor-parallel-size $parallel \
    --enforce-eager \
    --trust-remote-code &
    # --gpu-memory-utilization 0.9 \
    # --max-model-len 8192 \

    echo "run server on cuda $cuda, port $port "

done