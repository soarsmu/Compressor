```
CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --model_dir ../checkpoint \
    --size 3 \
    --model biGRU \
    --input_dim 208 \
    --hidden_dim 48 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/train_3_dis.log]
```

```
CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --model_dir ../checkpoint \
    --size 3 \
    --model biGRU \
    --input_dim 208 \
    --hidden_dim 48 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --epochs 15 \
    --seed 123456 2>&1| tee ../logs/eval_3_dis.log
```

```
python searcher.py \
    --population_size 20 \
    --generation_size 50 \
    --target_size 0.01 2>&1| tee ../logs/search.log
```