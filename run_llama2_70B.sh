export INTERNLM_ACCELERATOR=cuda

#export NCCL_SOCKET_IFNAME=bond0 #ens2f0np0
#export GLOO_SOCKET_IFNAME=bond0 #ens2f0np0
export NCCL_SOCKET_IFNAME=eth0 #ens2f0np0
export GLOO_SOCKET_IFNAME=eth0 #ens2f0np0
#export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
#export NCCL_IB_GID_INDEX=3
#export NCCL_IB_TC=160

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export ENABLE_FLASH_ATTENTION_WITH_IXDNN=1
export NCCL_SHM_DISABLE=0
export NCCL_USE_DIRECT=1
export NCCL_PRIM_P2P_LEVEL=2
export NCCL_USE_HIGHPRIORITYWARP=1
export NCCL_FORCESYNC_DISABLE=1

export TZ=UTC-8
log_file="llama2_70B_$(date +%Y%m%d_%H%M%S)"
# torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=9999  train.py --config configs/7B_llama2.py --launcher torch --backend nccl  2>&1 | tee -a llama2.log
# python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=9999  train.py --config 70B_llama2.py --launcher torch --backend nccl --profiling  2>&1 | tee -a llama2_70B.log
python3 -m torch.distributed.run --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=9999  train.py --config 70B_llama2.py --launcher torch --backend nccl 2>&1 | tee -a ${log_file}.log


