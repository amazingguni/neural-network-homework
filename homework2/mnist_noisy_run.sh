
source common.sh

export CUDA_VISIBLE_DEVICES="2"
DATA=mnist
TASK=3_noisy
USE_ALL=--use-all
LOSS="sce"
EPOCH=300
SUB_CKPT="./output/mnist/3_noisy/191205-1240/best.pth"
CONFIDENCE="0.98"
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
    --revise-label \
    --sub-ckpt $SUB_CKPT \
    --confidence $CONFIDENCE \
    $USE_ALL $BALANED
