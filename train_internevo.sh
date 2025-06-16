#!/bin/bash

# # 设置工作目录
# cd /container/cyx/

# # 启动 NPU 功率监控脚本
# nohup python npu_power.py &

# 设置日志文件名
log_file="llava_internevo_$(date +%Y%m%d_%H%M%S).log"

# 设置环境变量
export TZ=UTC-8
export MASTER_ADDR=localhost
export MASTER_PORT=8888
export WORLD_SIZE=8
export RANK=0
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_RDMA_TC=104
export HCCL_BUFFSIZE=2048
export CUDA_DEVICE_MAX_CONNECTIONS=8

# 加载 Ascend 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置 Python 路径
export PYTHONPATH=/container/cyx/deeplink.framework/dipu/:$PYTHONPATH
export PYTHONPATH=/container/cyx/DeepLinkExt/ditorch/:$PYTHONPATH
export PYTHONPATH=/container/cyx/DeepLinkExt:$PYTHONPATH

# 设置其他环境变量
export DIPU_DEVICE_MEMCACHING_ALGORITHM=TORCH
export DIPU_HOST_MEMCACHING_ALGORITHM=TORCH
export DIPU_TORCH_ALLOCATOR_CONF=expandable_segments:True
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export ASCEND_RUNTIME_PATH=/usr/local/Ascend/ascend-toolkit/latest/runtime
export INTERNLM_ACCELERATOR="ditorch"
export DEEPLINK_EXT_PLATFORM_TYPE=torch_npu
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 进入项目目录
cd /container/cyx/InternEvo

# 启动训练脚本
# torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=1 --node_rank=$RANK train.py --config configs/demo_llava_opt_sp1_nosave_v24.py --launcher torch --seed 42  2>&1 | tee logs/${log_file}
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 --nnodes=1 --node_rank=$RANK train.py --config /container/cyx/InternEvo/configs_test/demo_llava.py --launcher torch --seed 42  2>&1 | tee logs/${log_file}