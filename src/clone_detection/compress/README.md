
```
python searcher.py --size 3
```
Then you got the architecture for small models.

Try to train them.
```
CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --type unlabel_train \
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
    --seed 123456 2>&1| tee ../logs/unlabel_train_3.log


CUDA_VISIBLE_DEVICES=6 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 96 \
    --intermediate_size 64 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 1 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_unlabel_train_3.log


CUDA_VISIBLE_DEVICES=6 python d_bert.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../baseline \
    --size o_gcb \
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
    --seed 123456 2>&1| tee ../logs/baseline_gcb.log

CUDA_VISIBLE_DEVICES=6 python d_bert.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../baseline \
    --size o_gcb \
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
    --seed 123456 2>&1| tee ../logs/eval_baseline_gcb.log

CUDA_VISIBLE_DEVICES=4 python d_bert.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../baseline \
    --size n_gcb \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 200 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/n_baseline_gcb.log

CUDA_VISIBLE_DEVICES=4 python d_bert.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../baseline \
    --size n_gcb \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 200 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_n_baseline_gcb.log

CUDA_VISIBLE_DEVICES=6 python d_bert.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
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

CUDA_VISIBLE_DEVICES=0 python d_bert.py \
    --do_train \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --model_dir ../baseline \
    --size n \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 200 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/n_baseline.log

CUDA_VISIBLE_DEVICES=0 python d_bert.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/unlabel_train.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../baseline \
    --size n \
    --type unlabel_train \
    --attention_heads 8 \
    --hidden_dim 200 \
    --intermediate_size 64 \
    --n_layers 1 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/eval_n_baseline.log
```