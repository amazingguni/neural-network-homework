
source common.sh

export CUDA_VISIBLE_DEVICES="3"
DATA=mnist
TASK="2_semisupervised"
USE_ALL=--use-all
LOSS="sce"
EPOCH=250
CONFIDENCE=0.8
# Only Training set
#SEMI_CKPT="./output/mnist/2_semisupervised/191204-1331/best.pth"
# Use all
SEMI_CKPT="./output/mnist/2_semisupervised/191204-1423/best.pth"

python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    --loss $LOSS \
    --sce-alpha 0.1 \
    --sce-beta 1.0 \
    --semi  --semi-ckpt $SEMI_CKPT --confidence $CONFIDENCE \
    $USE_ALL $BALANED
