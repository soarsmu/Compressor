declare -a alphas=(1)

for loss in "ce"
do
  for alpha in ${alphas[@]}
  do
    MODEL_DIR=./checkpoint/Roberta/$loss/50/$alpha
    mkdir -p $MODEL_DIR
    CUDA_VISIBLE_DEVICES=7 python distill.py \
        --do_train \
        --train_data_file=../../../data/code_search/train.jsonl \
        --eval_data_file=../../../data/code_search/python_valid_0.jsonl \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --loss_func $loss \
        --alpha $alpha \
        --input_dim 512 \
        --hidden_dim 512 \
        --n_layers 2 \
        --vocab_size 10000 \
        --block_size 256 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --epochs 20 \
        --seed 123456 2>&1| tee $MODEL_DIR/train.log 

    CUDA_VISIBLE_DEVICES=7 python distill.py \
        --do_eval \
        --train_data_file=../../../data/code_search/train.jsonl \
        --eval_data_file=../../../data/code_search/test.jsonl \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --input_dim 512 \
        --hidden_dim 512 \
        --n_layers 2 \
        --vocab_size 10000 \
        --block_size 256 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --choice best \
        --seed 123456 2>&1| tee $MODEL_DIR/eval.log
  done
done
