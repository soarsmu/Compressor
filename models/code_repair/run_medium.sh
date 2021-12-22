mkdir -p logs/medium

CUDA_VISIBLE_DEVICES=0,6 python main.py \
    --do_train \
    --train_data_file=../../data/code_repair/medium/train.buggy-fixed.buggy,../../data/code_repair/medium/train.buggy-fixed.fixed \
    --eval_data_file=../../data/code_repair/medium/valid.buggy-fixed.buggy,../../data/code_repair/medium/valid.buggy-fixed.fixed \
    --output_dir ./medium \
    --epoch 40 \
    --beam_size 5 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee logs/medium/train.log

CUDA_VISIBLE_DEVICES=0,6 python main.py \
    --do_eval \
    --eval_data_file=../../data/code_repair/medium/test.buggy-fixed.buggy,../../data/code_repair/medium/test.buggy-fixed.fixed \
    --output_dir ./medium \
    --block_size 256 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee logs/medium/eval.log