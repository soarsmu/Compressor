CUDA_VISIBLE_DEVICES=6 python distill.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/BigCloneBench/train.txt \
    --eval_data_file=../../../data/clone_detection/BigCloneBench/valid.txt \
    --block_size 400 \
    --eval_batch_size 1 \
    --seed 123456 2>&1| tee logs/eval.log