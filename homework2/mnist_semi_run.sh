
source common.sh

DATA=mnist
TASK="2_semisupervised"
USE_ALL=--use-all
LOSS="sce"
EPOCH=200
CONFIDENCE=0.8
# Only Training set
#SEMI_CKPT="./output/mnist/2_semisupervised/191204-1331/best.pth"
# Use all
SUB_CKPT="./output/mnist/2_semisupervised/191204-1423/best.pth"
ALPHA=1.0
BETA=0.5

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
