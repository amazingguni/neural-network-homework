
source common.sh

export CUDA_VISIBLE_DEVICES="3"
DATA=mnist
TASK="2_semisupervised"
USE_ALL=--use-all
LOSS="ce"
EPOCH=300
python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    --loss $LOSS \
    --sce-alpha 0.2 \
    --sce-beta 1.0 \
    $USE_ALL $BALANED
