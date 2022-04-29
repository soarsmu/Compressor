mkdir -p ../logs

CUDA_VISIBLE_DEVICES=2,4 python main.py \
    --do_train \
    --train_data_file=../../../data/code_search/label.pkl \
    --eval_data_file=../../../data/code_search/valid.pkl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log

CUDA_VISIBLE_DEVICES=1 python main.py \
    --do_eval \
    --train_data_file=../../../data/code_search/label.pkl \
    --eval_data_file=../../../data/code_search/test.pkl \
    --block_size 400 \
    --eval_batch_size 50 \
    --seed 123456 2>&1| tee ../logs/eval.log