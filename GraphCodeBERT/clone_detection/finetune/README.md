We prepare the finetuned model here:
You can download it via:
```
wget https://smu-my.sharepoint.com/:u:/g/personal/jiekeshi_smu_edu_sg/EWntleLvrG1DhC5By_ryfLUBHwHn2KURPhzFDkZ10XujeA?download=1 -O ../checkpoint/model.bin
```
GraphCodeBERT need a parser to extract data flows from the source code, please go to `./parser` to compile the parser first. Pls run:
```
cd parser
bash build.sh
```
## Evaluation
For evaluating the fine-tuned model, please run:
```
mkdir -p ../logs
python3 main.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../../../data/clone_detection/label_train.txt \
    --eval_data_file=../../../data/clone_detection/test_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 4 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/eval.log
```
## Get soft labels
For getting soft labels to do knowledge distillation later, please run:
```
python3 main.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=../../../data/clone_detection/label_train.txt \
    --eval_data_file=../../../data/clone_detection/unlabel_train.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 4 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```
You will see the `preds_unlabel_train_gcb.npy` in `../../../data/clone_detection/`.

You will also see that the log outputs say the accuracy is 0. Don't worry,  `unlabel_train.txt` has no true labels, so the accuracy is not true.

## Finetuning
If you'd like to finetune a model from scratch, please run:
```
mkdir -p ../logs
python3 main.py \
    --output_dir=../checkpoint \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../data/clone_detection/label_train.txt \
    --eval_data_file=../../../data/clone_detection/valid_sampled.txt \
    --test_data_file=../../../data/clone_detection/test_sampled.txt \
    --epoch 3 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log
```
