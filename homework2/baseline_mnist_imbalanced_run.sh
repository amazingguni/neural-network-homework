
source common.sh
export CUDA_VISIBLE_DEVICES="0"

DATA=mnist
TASK=1_imbalanced
USE_ALL=--use-all
python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    $USE_ALL $BALANED
