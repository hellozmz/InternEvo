source /ailab_internevo_afs/zhumingzhu/env/ascend.env

#!/bin/bash

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

# cd /deeplink_afs/zhumingzhu/work/01perf/npu_power/
# nohup python npu_power.py &

# cp -rL /deeplink_afs/zhumingzhu/work/InternEvo /opt/InternEvo
cd /deeplink_afs/zhumingzhu/work/Monitor/InternEvo/

export TZ=UTC-8


export MASTER_ADDR=localhost
export MASTER_PORT=8888
export WORLD_SIZE=8
#  TODO: only one is 0........
export RANK=0


export TZ=UTC-8
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export INTERNLM_ACCELERATOR="ditorch"
export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# cp -rL /deeplink_afs/zhumingzhu/code/envs /opt/envs
export PYTHONPATH=/deeplink_afs/zhumingzhu/code/envs/rotary_emb_no_interleave/DeepLinkExt:/deeplink_afs/zhumingzhu/code/envs:$PYTHONPATH
apt-get install -y netcat

# torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=1 --node_rank=$RANK train.py --config configs/7B_sft.py --launcher torch --seed 42  2>&1 | tee logs/${log_file}.log


# Function to start torchrun
start_torchrun() {
    log_file="internevo_$(date +%Y%m%d_%H%M%S)"
    torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=$NNODES --node_rank=$RANK train.py --config configs/7B_sft.py --launcher torch --seed 42 2>&1 | tee logs/${log_file}.log &
    torchrun_pid=$!
    echo "torchrun started with PID: $torchrun_pid"
}

# Function to monitor and restart torchrun if needed
monitor() {
    while true; do
        cmd=$(nc -l -p 55555)
        if [ "$cmd" == "restart" ]; then
            echo "Restarting torchrun..."
            kill $torchrun_pid
            wait $torchrun_pid
            start_torchrun
        fi
        sleep 5
    done
}

# Start torchrun
start_torchrun

# Start monitoring only on rank 0
if [ "$RANK" -eq 0 ]; then
    monitor
fi
