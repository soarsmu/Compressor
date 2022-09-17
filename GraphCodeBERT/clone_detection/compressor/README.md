**Note**: before running the following scripts, please make sure soft labels `../../../data/clone_detection/preds_unlabel_train.npy` exist.

If it does not exist, please go to `../finetune/README.md` to see how to get it.

GraphCodeBERT need a parser to extract data flows from the source code, please go to `./parser` to compile the parser first. Pls run:
```
cd parser
bash build.sh
```

## 3 MB Model

In our paper, the architecture-related hyperparamenters for 3 MB is `{'attention_heads': 8, 'hidden_dim': 96, 'intermediate_size': 64, 'n_layers': 12, 'vocab_size': 1000}`.

We release a trained 3 MB model in `../checkpoint/3`. For evaluating this model, please run:
```
mkdir -p ../logs
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/eval_3.log
```

For training a 3 MB model from scratch, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train_3.log
```

## 25 MB Model

In our paper, the architecture-related hyperparamenters for 25 MB is `{'attention_heads': 16, 'hidden_dim': 432, 'intermediate_size': 128, 'n_layers': 6, 'vocab_size': 1000}`.

We release a trained 25 MB model in `../checkpoint/25`. For evaluating this model, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 25 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/eval_25.log
```

For training a 25 MB model from scratch, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 25 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 432 \
    --intermediate_size 128 \
    --n_layers 6 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train_25.log
```
## 50 MB Model

In our paper, the architecture-related hyperparamenters for 50 MB is `{'attention_heads': 16, 'hidden_dim': 480, 'intermediate_size': 576, 'n_layers': 6, 'vocab_size': 6000}`.

We release a trained 50 MB model in `../checkpoint/50`. For evaluating this model, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 50 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 480 \
    --intermediate_size 576 \
    --n_layers 6 \
    --vocab_size 6000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/eval_50.log
```
For training a 50 MB model from scratch, please run:
```
python3 distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 10 \
    --size 50 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 480 \
    --intermediate_size 576 \
    --n_layers 6 \
    --vocab_size 6000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train_50.log
```
## Baseline Model

For evaluating the existing LSTM baseline model, please run:
```
python3 lstm_baseline.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --model_dir ../checkpoint \
    --hidden_dim 300 \
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
python3 lstm_baseline.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --model_dir ../checkpoint \
    --hidden_dim 300 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/train_baseline.log
```