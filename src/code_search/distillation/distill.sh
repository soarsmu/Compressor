declare -a alphas=(1.0 0.9)

for loss in "ce"
do
  for alpha in ${alphas[@]}
  do
    MODEL_DIR=./checkpoint//$loss/50/$alpha
    mkdir -p $MODEL_DIR
    CUDA_VISIBLE_DEVICES=0 python distill.py \
        --do_train \
        --train_data_file=../../../data/code_search/train.jsonl \
        --eval_data_file=../../../data/code_search/valid.jsonl \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --loss_func $loss \
        --alpha $alpha \
        --input_dim 256 \
        --hidden_dim 384 \
        --n_layers 4 \
        --vocab_size 14000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --epochs 15 \
        --seed 123456 2>&1| tee $MODEL_DIR/train.log 

    CUDA_VISIBLE_DEVICES=2 python distill.py \
        --do_eval \
        --train_data_file=../../../data/code_search/train.jsonl \
        --eval_data_file=../../../data/code_search/test.jsonl \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --input_dim 256 \
        --hidden_dim 384 \
        --n_layers 4 \
        --vocab_size 14000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --choice best \
        --seed 123456 2>&1| tee $MODEL_DIR/eval.log
  done
done 