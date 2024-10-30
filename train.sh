export MASTER_ADDR=localhost
export MASTER_PORT=8888
export WORLD_SIZE=1
export RANK=0


export TZ=UTC-8
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export INTERNLM_ACCELERATOR="npu"

log_file="llava_$(date +%Y%m%d_%H%M%S)"

torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK train.py --config configs/demo_llava.py --launcher torch --seed 1024 