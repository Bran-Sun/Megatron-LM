#! /bin/bash

# 1.3B model
GLOBAL_BATCH_SIZE=64
NUM_MICRO_BATCH=8
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_HEADS=32
RESULT_DIR="end_to_end_result"
MASTER_ADDR="nico1"

LOCAL_BATCH_SIZE=$(expr $GLOBAL_BATCH_SIZE / $NUM_MICRO_BATCH)

num=8

TP=1
while [ $TP -le $num ]
do
    LEFT=$(expr $num / $TP)
    DP=1
    while [ $DP -le $LEFT ]
    do
        PP=$(expr $LEFT / $DP)
        MICRO_BATCH_SIZE=$(expr $LOCAL_BATCH_SIZE / $DP)
        echo "test DP: $DP, TP: $TP, PP: $PP"

        TP=$TP PP=$PP \
        GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
        NUM_LAYERS=$NUM_LAYERS HIDDEN_SIZE=$HIDDEN_SIZE NUM_HEADS=$NUM_HEADS \
        MASTER_ADDR=$MASTER_ADDR \
        srun -n 8 -p Big -K -w nico1 sh examples/pretrain_gpt.sh | tee ${RESULT_DIR}/gpt_${TP}_${DP}_${PP}.txt

        # openmpi
        # mpirun --allow-run-as-root -x TP=$TP -x DP=$PP \
        # -x GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE -x MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE \
        # -x NUM_LAYERS=$NUM_LAYERS -x HIDDEN_SIZE=$HIDDEN_SIZE -x NUM_HEADS=$NUM_HEADS \
        # -x LD_LIBRARY_PATH -x MASTER_ADDR=$MASTER_ADDR \
        # sh examples/pretrain_gpt.sh | tee ${RESULT_DIR}/gpt_${TP}_${DP}_${PP}.txt

        DP=$(expr $DP \* 2)
    done
    TP=$(expr $TP \* 2)
done
