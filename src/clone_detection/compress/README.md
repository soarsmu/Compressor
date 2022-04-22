# GRU 50MB: 256 384 4 14000
# GRU 20MB: 256 256 3 8000
# GRU 3MB: 128 256 1 2000
# TODO: LSTM and CNN
declare -a alphas=(1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.15 0.1 0.05 0.0)

declare -a alphas=(1.0)

for loss in "ce"
do
  for alpha in ${alphas[@]}
  do
    MODEL_DIR=./checkpoint/Roberta/$loss/50/$alpha
    mkdir -p $MODEL_DIR
    CUDA_VISIBLE_DEVICES=0 python distill.py \
        --do_train \
        --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
        --eval_data_file=../../../../data/clone_detection/BigCloneBench/valid.txt \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --loss_func $loss \
        --alpha $alpha \
        --input_dim 512 \
        --hidden_dim 768 \
        --n_layers 2 \
        --vocab_size 14000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --epochs 25 \
        --seed 123456 2>&1| tee $MODEL_DIR/train.log 

    CUDA_VISIBLE_DEVICES=2 python distill.py \
        --do_eval \
        --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
        --eval_data_file=../../../../data/clone_detection/BigCloneBench/test.txt \
        --model_dir $MODEL_DIR \
        --std_model Roberta \
        --input_dim 512 \
        --hidden_dim 768 \
        --n_layers 2 \
        --vocab_size 14000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --choice best \
        --seed 123456 2>&1| tee $MODEL_DIR/eval.log 
  done
done 

declare -a alphas=(0.5 1.0 0.9)

for loss in "ce"
do
  for alpha in ${alphas[@]}
  do
    MODEL_DIR=./checkpoint//$loss/20/$alpha
    mkdir -p $MODEL_DIR
    # CUDA_VISIBLE_DEVICES=6 python distill.py \
    #     --do_train \
    #     --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
    #     --eval_data_file=../../../../data/clone_detection/BigCloneBench/valid.txt \
    #     --model_dir $MODEL_DIR \
    #     --std_model biGRU \
    #     --loss_func $loss \
    #     --alpha $alpha \
    #     --input_dim 256 \
    #     --hidden_dim 256 \
    #     --n_layers 3 \
    #     --vocab_size 8000 \
    #     --block_size 400 \
    #     --train_batch_size 16 \
    #     --eval_batch_size 64 \
    #     --epochs 25 \
    #     --seed 123456 2>&1| tee $MODEL_DIR/train.log &

    CUDA_VISIBLE_DEVICES=4 python distill.py \
        --do_eval \
        --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
        --eval_data_file=../../../../data/clone_detection/BigCloneBench/test.txt \
        --model_dir $MODEL_DIR \
        --std_model biGRU \
        --input_dim 256 \
        --hidden_dim 256 \
        --n_layers 3 \
        --vocab_size 8000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --choice best \
        --seed 123456 2>&1| tee $MODEL_DIR/eval.log &
  done
done 

declare -a alphas=(0.5 1.0 0.9)

for loss in "ce"
do
  for alpha in ${alphas[@]}
  do
    MODEL_DIR=./checkpoint//$loss/3/$alpha
    mkdir -p $MODEL_DIR
    # CUDA_VISIBLE_DEVICES=7 python distill.py \
    #     --do_train \
    #     --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
    #     --eval_data_file=../../../../data/clone_detection/BigCloneBench/valid.txt \
    #     --model_dir $MODEL_DIR \
    #     --std_model biGRU \
    #     --loss_func $loss \
    #     --alpha $alpha \
    #     --input_dim 128 \
    #     --hidden_dim 256 \
    #     --n_layers 1 \
    #     --vocab_size 2000 \
    #     --block_size 400 \
    #     --train_batch_size 16 \
    #     --eval_batch_size 64 \
    #     --epochs 25 \
    #     --seed 123456 2>&1| tee $MODEL_DIR/train.log &

    CUDA_VISIBLE_DEVICES=5 python distill.py \
        --do_eval \
        --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
        --eval_data_file=../../../../data/clone_detection/BigCloneBench/test.txt \
        --model_dir $MODEL_DIR \
        --std_model biGRU \
        --input_dim 128 \
        --hidden_dim 256 \
        --n_layers 1 \
        --vocab_size 2000 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 64 \
        --choice best \
        --seed 123456 2>&1| tee $MODEL_DIR/eval.log &
  done
done 


CUDA_VISIBLE_DEVICES=1 python distillation.py \
    --do_train \
    --train_data_file=../../../data/clone_search/valid_sampled.txt \
    --eval_data_file=../../../data/clone_search/test_sampled.txt \
    --model_dir ../checkpoint \
    --size 3 \
    --model Transformer \
    --input_dim 208 \
    --hidden_dim 48 \
    --n_layers 12 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 1 \
    --eval_batch_size 64 \
    --epochs 15 \
    --seed 123456