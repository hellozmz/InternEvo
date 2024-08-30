export INTERNLM_ACCELERATOR=cuda


export TZ=UTC-8
log_file="llama2_7B_$(date +%Y%m%d_%H%M%S)"

# torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=9999  train.py --config configs/7B_llama2.py --launcher torch --backend nccl  2>&1 | tee -a llama2.log
python3 -m torch.distributed.run --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=9999  train.py --config configs/7B_llama2.py --launcher torch --backend nccl  2>&1 | tee -a tee -a ${log_file}.log


