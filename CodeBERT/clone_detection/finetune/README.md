## Finetuning
We prepare the finetuned model here:
You can download it via:
```
wget https://smu-my.sharepoint.com/:u:/g/personal/jiekeshi_smu_edu_sg/EdGD1SxoWqRDt0dVcqNO0TwBAluQj0KJA2DE9swMwUHkng?download=1 -O ../checkpoint/model.bin
```
If you'd like to finetune a model from scratch, please run:
```
mkdir -p ../logs
python3 main.py \
    --do_train \
    --train_data_file=../../../data/clone_detection/train_sampled.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log
```

## Evaluation
For evaluating the fine-tuned model, please run:
```
python3 main.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/train_sampled.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/eval.log
```

## Get soft labels
For getting soft labels to do knowledge distillation later, please run:
```
CUDA_VISIBLE_DEVICES=4 python3 main.py \
    --do_eval \
    --train_data_file=../../../data/clone_detection/train_sampled.txt \
    --eval_data_file=../../../data/clone_detection/unlabel_train.txt \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456
```
You will see the `preds_unlabel_train.npy` in `../../../data/clone_detection/`. 

You will also see that the log outputs say the accuracy is 0. Don't worry,  `unlabel_train.txt` has no true labels, so the accuracy is not true.