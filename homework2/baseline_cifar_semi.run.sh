
source common.sh
export CUDA_VISIBLE_DEVICES="3"

DATA=cifar
TASK=2_semisupervised
USE_ALL=--use-all
EPOCH=250
python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    $USE_ALL $BALANED
