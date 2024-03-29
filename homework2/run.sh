
source common.sh

export CUDA_VISIBLE_DEVICES="0"
DATA=cifar
TASK=origin
#USE_ALL='--use-all'
EPOCH=1
BALANED='--balanced'

python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    $USE_ALL $BALANED
