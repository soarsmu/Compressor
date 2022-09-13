## Finetuning
We prepare the finetuned model here:
You can download it via:
```

```
If you'd like to finetune a model from scratch, please run:
```
mkdir -p ../logs
CUDA_VISIBLE_DEVICES=2 python3 main.py \
    --do_train \
    --train_data_file=../../../data/clone_search/valid_sampled.txt \
    --eval_data_file=../../../data/clone_search/valid_sampled.txt \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 2 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log
```

## Evaluation
For evaluating the 
```
CUDA_VISIBLE_DEVICES=4 python3 main.py \
    --do_eval \
    --train_data_file=../../../data/clone_search/train_sampled.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --block_size 400 \
    --eval_batch_size 1 \
    --seed 123456 2>&1| tee ../logs/eval.log
```

###