
source common.sh

export CUDA_VISIBLE_DEVICES="1"
DATA=cifar
TASK=1_imbalanced
BALANED='--balanced'
USE_ALL='--use-all'

python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    $USE_ALL $BALANED
