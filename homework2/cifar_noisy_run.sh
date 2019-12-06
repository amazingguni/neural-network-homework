
source common.sh

DATA=cifar
TASK=3_noisy
USE_ALL=--use-all
LOSS="sce"
SUB_CKPT="./output/cifar/3_noisy/191205-1239/best.pth"
CONFIDENCE="0.98"
ALPHA="1.0"
BETA="0.5"

python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    --loss $LOSS \
    --sce-alpha $ALPHA \
    --sce-beta $BETA \
    --revise-label \
    --sub-ckpt $SUB_CKPT \
    --confidence $CONFIDENCE \
    $USE_ALL $BALANED
