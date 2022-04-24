mkdir logs

CUDA_VISIBLE_DEVICES=3 python main.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log

CUDA_VISIBLE_DEVICES=3 python main.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/eval.log