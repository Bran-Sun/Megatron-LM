#! /bin/bash

# 1.3B model
GLOBAL_BATCH_SIZE=512
NUM_MICRO_BATCH=64

NUM_LAYERS=2
HIDDEN_SIZE=4096
NUM_HEADS=32
RESULT_DIR="intra_stage_result_2"
MASTER_ADDR="nico1"

LOCAL_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $NUM_MICRO_BATCH)

num=2

if [ ! -d $RESULT_DIR ]; then
  mkdir $RESULT_DIR
fi

TP=1
PP=1
while [ $TP -le $num ]
do
    DP=$(expr $num / $TP)
    MICRO_BATCH_SIZE=$(expr $LOCAL_BATCH_SIZE / $DP)
    echo "test DP: $DP, TP: $TP, PP: $PP"

    TP=$TP PP=$PP \
    GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
    NUM_LAYERS=$NUM_LAYERS HIDDEN_SIZE=$HIDDEN_SIZE NUM_HEADS=$NUM_HEADS \
    MASTER_ADDR=$MASTER_ADDR \
    srun -n $num -p Big -K -w nico1 sh examples/pretrain_gpt.sh | tee ${RESULT_DIR}/gpt_${TP}_${DP}_${PP}.txt

    # openmpi
    # mpirun --allow-run-as-root -x TP=$TP -x DP=$PP \
    # -x GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE -x MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
    # -x NUM_LAYERS=$NUM_LAYERS -x HIDDEN_SIZE=$HIDDEN_SIZE -x NUM_HEADS=$NUM_HEADS \
    # -x LD_LIBRARY_PATH -x MASTER_ADDR=$MASTER_ADDR \
    # sh examples/pretrain_gpt.sh | tee ${RESULT_DIR}/gpt_${TP}_${DP}_${PP}.txt

    TP=$(expr $TP \* 2)
done
