CUDA_VISIBLE_DEVICES=3 python distill.py \
    --do_train \
    --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
    --eval_data_file=../../../../data/clone_detection/BigCloneBench/valid.txt \
    --std_model Roberta \
    --vocab_size 10000 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --epochs 5 \
    --seed 123456 2>&1| tee ../logs/dis.log

CUDA_VISIBLE_DEVICES=3 python distill.py \
    --do_eval \
    --train_data_file=../../../../data/clone_detection/BigCloneBench/train.txt \
    --eval_data_file=../../../../data/clone_detection/BigCloneBench/test.txt \
    --std_model Roberta \
    --vocab_size 10000 \
    --block_size 400 \
    --eval_batch_size 64 \
    --choice best \
    --seed 123456 2>&1| tee ../logs/dis_eval.log