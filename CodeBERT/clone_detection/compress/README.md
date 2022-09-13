
**Note**: before running the following scripts, please make sure `../../../data/clone_detection/preds_unlabel_train.npy` exist.

If it does not exist, please go to `../finetune/README.md` to see how to get it.

## 3 MB Model

In our paper, the architecture-related hyperparamenters for 3 MB is `{'attention_heads': 8, 'hidden_dim': 96, 'intermediate_size': 64, 'n_layers': 12, 'vocab_size': 1000}`.

We release a trained 3 MB model in `../checkpoint/3`. For evaluating this model, please run:
```
python3 distill.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_3.log
```

For training a 3 MB model from scratch, please run:
```
python3 distill.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/train_3.log
```

## 25 MB Model

In our paper, the architecture-related hyperparamenters for 25 MB is `{'attention_heads': 16, 'hidden_dim': 432, 'intermediate_size': 128, 'n_layers': 6, 'vocab_size': 1000}`.

We release a trained 25 MB model in `../checkpoint/25`. For evaluating this model, please run:
```
CUDA_VISIBLE_DEVICES=2 python3 distill.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 25 \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_25.log
```

For training a 25 MB model from scratch, please run:
```
python3 distill.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 25 \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/train_25.log
```

## 50 MB Model

In our paper, the architecture-related hyperparamenters for 50 MB is `{'attention_heads': 16, 'hidden_dim': 480, 'intermediate_size': 576, 'n_layers': 6, 'vocab_size': 6000}`.

We release a trained 50 MB model in `../checkpoint/50`. For evaluating this model, please run:
```
CUDA_VISIBLE_DEVICES=2 python3 distill.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 50 \
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
    --seed 123456 2>&1| tee ../logs/eval_50.log
```

For training a 50 MB model from scratch, please run:
```
python3 distill.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 50 \
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
    --seed 123456 2>&1| tee ../logs/train_50.log
```

## Baseline Model

For evaluating the existing LSTM baseline model, please run:
```
CUDA_VISIBLE_DEVICES=6 python3 lstm_baseline.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --model_dir ../baseline \
    --size o \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 300 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_baseline.log
```

For training the LSTM baseline model from scratch, please run:
```
CUDA_VISIBLE_DEVICES=6 python3 lstm_baseline.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --model_dir ../baseline \
    --size o \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 300 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/train_baseline.log
```