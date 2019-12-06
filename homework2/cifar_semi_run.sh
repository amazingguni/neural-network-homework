
source common.sh

DATA=cifar
TASK="2_semisupervised"
USE_ALL=--use-all
LOSS="sce"
#LOSS="ce"
CONFIDENCE=0.8
# Only Training set
# SEMI_CKPT="./output/cifar/2_semisupervised/191204-1437/best.pth"
# Use all
SUB_CKPT="./output/cifar/2_semisupervised/191204-1438/best.pth"
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
    --semi  --sub-ckpt $SUB_CKPT --confidence $CONFIDENCE \
    $USE_ALL $BALANED
