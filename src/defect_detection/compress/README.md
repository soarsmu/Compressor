```
CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 3 \
    --model LSTM \
    --input_dim 128 \
    --hidden_dim 80 \
    --n_layers 10 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/train_3_dis.log
```

CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 3 \
    --model biGRU \
    --input_dim 128 \
    --hidden_dim 80 \
    --n_layers 10 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice recent_mix \
    --seed 123456 2>&1| tee ../logs/eval_3_dis.log

```
## 0.01 label
CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_train_001.log

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_train_005.log &

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_label_train_005.log

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_train_01.log &

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_label_train_01.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/eval_label_train_001.log

## 0.01 unlabel

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_001.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_001.log

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --eval_batch_size 64 \
    --learning_rate 1e-3 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_005.log & 

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_005.log

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/unlabel_train_01.log & 

CUDA_VISIBLE_DEVICES=4 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_01.log

## 0.01 mix
CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_train_001.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_train_005.log &

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_mixed_train_005.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_train_01.log &

CUDA_VISIBLE_DEVICES=2 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_mixed_train_01.log


## 0.01 label_dis
CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_dis_001.log


CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_dis_005.log &


CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_label_dis_005.log

CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/label_dis_01.log &

CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_label_dis_01.log

## 0.01 mixed_dis
CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_001.log

CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_005.log & 

CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_mixed_dis_005.log

CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
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
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mixed_dis_01.log &

CUDA_VISIBLE_DEVICES=7 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
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
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_mixed_dis_01.log

CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.01 \
    --model biGRU \
    --input_dim 128 \
    --hidden_dim 80 \
    --n_layers 10 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_dis_001.log

CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.01 \
    --model biGRU \
    --input_dim 128 \
    --hidden_dim 80 \
    --n_layers 10 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --epochs 15 \
    --choice best_un \
    --seed 123456 2>&1| tee ../logs/un_eval_dis_001.log
```

```
python searcher.py \
    --population_size 20 \
    --generation_size 20 \
    --target_size 0.1 2>&1| tee ../logs/search.log
```
0.01 {'hidden_dim': 80, 'input_dim': 128, 'model_arch': 'biGRU', 'n_layers': 10, 'vocab_size': 1000}
0.05 {'hidden_dim': 272, 'input_dim': 48, 'model_arch': 'biLSTM', 'n_layers': 4, 'vocab_size': 2000}
0.1 {'hidden_dim': 432, 'input_dim': 192, 'model_arch': 'biLSTM', 'n_layers': 3, 'vocab_size': 7000}

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/mixed_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/mix_train_dis_005.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/train_dis_005.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/un_train_dis_005.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice best \
    --seed 123456 2>&1| tee ../logs/eval_dis_005.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice best_un \
    --seed 123456 2>&1| tee ../logs/un_eval_dis_005.log

CUDA_VISIBLE_DEVICES=0 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.05 \
    --model biLSTM \
    --input_dim 48 \
    --hidden_dim 272 \
    --n_layers 4 \
    --vocab_size 2000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice best_mix \
    --seed 123456 2>&1| tee ../logs/mix_eval_dis_005.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 0.1 \
    --model biLSTM \
    --input_dim 192 \
    --hidden_dim 432 \
    --n_layers 3 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/train_dis_01.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 0.1 \
    --model biLSTM \
    --input_dim 192 \
    --hidden_dim 432 \
    --n_layers 3 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/un_train_dis_01.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/label_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.1 \
    --model biLSTM \
    --input_dim 192 \
    --hidden_dim 432 \
    --n_layers 3 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/eval_dis_01.log

CUDA_VISIBLE_DEVICES=3 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/unlabel_train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 0.1 \
    --model biLSTM \
    --input_dim 192 \
    --hidden_dim 432 \
    --n_layers 3 \
    --vocab_size 7000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 15 \
    --choice recent_un \
    --seed 123456 2>&1| tee ../logs/un_eval_dis_01.log