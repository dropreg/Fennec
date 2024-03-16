#!/bin/bash

MODEl_NAME=/public/home/ljt/lxb/hf_lfs/zephyr-7b-beta
# MODEl_NAME=/public/home/ljt/hf_models/Llama-2-13b-chat-hf
# MODEl_NAME=/public/home/ljt/hf_models/Llama-2-7b-chat-hf
# MODEl_NAME=/public/home/ljt/lxb/hf_lfs/Mistral-7B-Instruct-v0.2

EVAL_PARALLEL=true

echo "run vllm server parallel..."

cuda_list=(0 1 2 3 4 5 6 7)
port_list=(8000 8001 8002 8003 8004 8005 8006 8007)

if [ "$EVAL_PARALLEL" == "true" ]; then
    length=${#cuda_list[@]}
else
    length=1
fi

for ((i=0; i<$length; i++)); do
    cuda=${cuda_list[$i]}
    port=${port_list[$i]}

    CUDA_VISIBLE_DEVICES=$cuda python -m vllm.entrypoints.api_server \
    --port $port \
    --model $MODEl_NAME \
    --tokenizer $MODEl_NAME \
    --trust-remote-code &

    echo "run server on cuda $cuda, port $port "

done