mkdir logs

CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/POJ-104/train.jsonl \
    --eval_data_file=../../../data/clone_detection/POJ-104/valid.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee logs/train.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/POJ-104/train.jsonl \
    --eval_data_file=../../../data/clone_detection/POJ-104/valid.jsonl \
    --block_size 400 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee logs/eval.log