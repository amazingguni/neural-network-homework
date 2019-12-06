
source common.sh

DATA=mnist
TASK=3_noisy
USE_ALL=--use-all
LOSS="sce"
SUB_CKPT="./output/mnist/3_noisy/191205-1240/best.pth"
CONFIDENCE="0.98"
EPOCH=300
ALPHA=0.1
BETA=1.0

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
