CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 176 \
    --intermediate_size 64 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_001.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 176 \
    --intermediate_size 64 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_001.log


CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 14 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_005.log


CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 14 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_005.log

CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 480 \
    --intermediate_size 576 \
    --n_layers 6 \
    --vocab_size 6000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_01.log

CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 480 \
    --intermediate_size 576 \
    --n_layers 6 \
    --vocab_size 6000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_01.log