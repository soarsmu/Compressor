# GRU 50MB: 256 384 4 14000
# GRU 20MB: 256 256 3 8000
# GRU 3MB: 128 256 1 2000
# TODO: LSTM 



## label
CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type label_train \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_train_001.log &

CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type label_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_train_005.log &

CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type label_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_train_01.log &



## unlabel

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type unlabel_train \
    --model biGRU \
    --input_dim 432 \
    --hidden_dim 32 \
    --n_layers 5 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_001.log


CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type unlabel_train \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_001.log &

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_mrr.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type unlabel_train \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 10 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --choice best \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_001.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type unlabel_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 2 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_005.log &

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_mrr.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type unlabel_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 5 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_005.log

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type unlabel_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_01.log &

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_mrr.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type unlabel_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 10 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_01.log

## mix
CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type mixed_train \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_train_001.log &

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type mixed_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_train_005.log & 

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type mixed_train \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_train_01.log &

## label_dis
CUDA_VISIBLE_DEVICES=5 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type label_dis \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_dis_001.log &

CUDA_VISIBLE_DEVICES=5 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type label_dis \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_dis_005.log & 

CUDA_VISIBLE_DEVICES=5 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type label_dis \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/label_dis_01.log &


###### mixed_dis
CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.01 \
    --type mixed_dis \
    --model biGRU \
    --input_dim 16 \
    --hidden_dim 112 \
    --n_layers 6 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_001.log &

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.05 \
    --type mixed_dis \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_005.log &

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/mixed_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 0.1 \
    --type mixed_dis \
    --model biLSTM \
    --input_dim 32 \
    --hidden_dim 464 \
    --n_layers 3 \
    --vocab_size 8000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_01.log &