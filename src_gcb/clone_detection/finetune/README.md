mkdir -p ../logs
CUDA_VISIBLE_DEVICES=5,0 python main.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 3 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log


CUDA_VISIBLE_DEVICES=5 python main.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../../data/clone_search/label_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/unlabel_train.txt \
    --epoch 4 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/eval.log


CUDA_VISIBLE_DEVICES=4 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
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
    --seed 123456  2>&1 | tee ../logs/distill_3.log

CUDA_VISIBLE_DEVICES=0 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
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
    --eval_batch_size 1 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/eval_distill_3.log


CUDA_VISIBLE_DEVICES=0 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 10 \
    --size 0.01 \
    --type unlabel_train \
    --attention_heads 16 \
    --hidden_dim 176 \
    --intermediate_size 64 \
    --n_layers 6 \
    --vocab_size 1000 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/eval_distill_0.01.log


CUDA_VISIBLE_DEVICES=4 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 10 \
    --size 0.05 \
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
    --seed 123456  2>&1 | tee ../logs/distill_0.05.log


CUDA_VISIBLE_DEVICES=4 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 10 \
    --size 0.05 \
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
    --seed 123456  2>&1 | tee ../logs/eval_distill_0.05.log


CUDA_VISIBLE_DEVICES=5 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 10 \
    --size 0.1 \
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
    --seed 123456  2>&1 | tee ../logs/distill_0.1.log


CUDA_VISIBLE_DEVICES=5 python distill.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --test_data_file=../../../data/clone_search/test_sampled.txt \
    --epoch 10 \
    --size 0.1 \
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
    --seed 123456  2>&1 | tee ../logs/eval_distill_0.1.log