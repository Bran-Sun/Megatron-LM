#! /bin/bash

# PLEASE input TP, PP, DP, LOCAL_BATCH_SIZE, NUM_LAYERS, HIDDEN_SIZE, NUM_HEADS

DATA_PATH=/home/sunzhenbo/dataset/dataset/my-bert_text_sentence

export MASTER_ADDR=nico1
export MASTER_PORT=6000

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export RANK=$SLURM_PROCID
export LOCAL_RANK=$(expr $RANK % $GPUS_PER_NODE)
export WORLD_SIZE=$SLURM_NTASKS

python pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --timing-log-level 2 \
       --timing-log-option all \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE  \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 20 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 94,5,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 4 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 0 \
       --make-vocab-size-divisible-by 51200
