#!/bin/bash

source /deeplink_afs/zhumingzhu/work/Monitor/ascend.env

export TZ=UTC-8
__conda_setup="$('/opt/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate torch_npu_py39

export RANK=1
export HCCL_CONNECT_TIMEOUT=120

stop_process() {
    torchrun_pid=$(pgrep -f "torchrun")
    if [ -n "$torchrun_pid" ]; then
        echo "Stopping running torchrun process with PID: $torchrun_pid"
        kill -9 $torchrun_pid
        wait $torchrun_pid
        sleep 15
    fi

    python_pid=$(pgrep -f "train.py --config configs/7B_sft.py --launcher torch --seed 42")
    if [ -n "$python_pid" ]; then
        echo "Stopping running Python process with PID: $python_pid"
        kill -9 $python_pid
        wait $python_pid
        sleep 15
    fi
}

stop_process

master_port=$1

torchrun --master_addr=10.119.40.90 --master_port=$master_port --nproc_per_node=2 --nnodes=2 --node_rank=1 train.py --config configs/7B_sft.py --launcher torch --seed 42 2>&1 | tee test_env1.log