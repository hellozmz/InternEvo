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


export RANK=0
export HCCL_CONNECT_TIMEOUT=120

iplist=("10.119.40.90" "10.119.45.60")
cmd_remote="cd /deeplink_afs/zhumingzhu/work/Monitor/InternEvo && bash start1.sh"


find_available_port() {
    while true; do
        port=$(shuf -i 30000-50000 -n 1)
        if ! nc -z 127.0.0.1 $port; then
            echo $port
            return
        fi
    done
}

execute_remote() {
    remote_ip=${iplist[1]}
    remote_command="$cmd_remote $master_port"

    ssh -o StrictHostKeyChecking=no $remote_ip "$remote_command"
}

start_torchrun() {
    log_file="internevo_$(date +%Y%m%d_%H%M%S)_${RANK}"
    master_port=$(find_available_port)
    echo "==================Starting torchrun with master_port: $master_port=================="
    # torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=2 --nnodes=$WORLD_SIZE --node_rank=$RANK train.py --config configs/7B_sft.py --launcher torch --seed 42 2>&1 | tee logs/${log_file}.log
    # torchrun --master_addr=10.119.40.90 --master_port=$master_port --nproc_per_node=2 --nnodes=2 --node_rank=0 train.py --config configs/7B_sft.py --launcher torch --seed 42 2>&1 | tee logs/${log_file}.log &
    torchrun --master_addr=${iplist[0]} --master_port=$master_port --nproc_per_node=2 --nnodes=2 --node_rank=0 train.py --config configs/7B_sft.py --launcher torch --seed 42 2>&1 | tee test_env0.log &
    torchrun_pid=$!
    echo "==================torchrun started with PID: $torchrun_pid=================="
    execute_remote
}

# Function to monitor and restart torchrun if needed
monitor() {
    while true; do
        cmd=$(nc -l -p 60006)
        if [ "$cmd" == "restart" ]; then
            echo "Restarting torchrun..."
            # kill $torchrun_pid
            # wait $torchrun_pid
            stop_process
            sleep 10
            echo "==================restart start_torchrun=================="
            start_torchrun &
            echo "==================finish restart start_torchrun=================="
            # execute_remote
        fi
        sleep 10
    done
}


stop_process() {
    torchrun_pid=$(pgrep -f "torchrun --master_addr")
    if [ -n "$torchrun_pid" ]; then
        echo "==================Stopping running torchrun process with PID: $torchrun_pid=================="
        kill -9 $torchrun_pid
        wait $torchrun_pid
        sleep 10
    fi

    python_pid=$(pgrep -f "train.py --config configs/7B_sft.py --launcher torch --seed 42")
    if [ -n "$python_pid" ]; then
        echo "==================Stopping running Python process with PID: $python_pid=================="
        kill -9 $python_pid
        wait $python_pid
        sleep 10
    fi
}

# Function to stop previous instances of start0.sh
stop_previous_start0() {
    start0_pid=$(pgrep -f "start0.sh")
    current_pid=$$
    for pid in $start0_pid; do
        if [ "$pid" != "$current_pid" ]; then
            echo "==================Stopping previous start0.sh process with PID: $pid=================="
            kill -9 $pid
            # wait $pid
            sleep 5
        fi
    done
}

# Check and stop any existing start0.sh processes
stop_previous_start0

stop_process


# 删除之前的数据
rm -rf llm_ckpts/*

echo "==================start_torchrun=================="
# Start torchrun
start_torchrun &
echo "==================finish start_torchrun=================="

# Start monitoring only on rank 0
if [ "$RANK" -eq 0 ]; then
    # apt-get install -y netcat
    echo "==================start monitor=================="
    monitor &
    echo "==================finish start monitor=================="
fi
