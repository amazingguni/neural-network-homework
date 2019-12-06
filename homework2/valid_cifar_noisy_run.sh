source common.sh

export CUDA_VISIBLE_DEVICES="3"
DATA=cifar
TASK=3_noisy

python Answer.py \
    --epoch $EPOCH \
    --lr $LR \
    --data $DATA \
    --task $TASK \
    --print 1 \
    --batch-size $BATCH_SIZE \
    $USE_ALL $BALANED --only-valid
