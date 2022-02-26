CUDA_VISIBLE_DEVICES=7 python distill.py \
    --do_train \
    --train_data_file=../../../data/defect_detection/train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --std_model biGRU \
    --vocab_size 10000 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --epochs 10 \
    --seed 123456 2>&1| tee ../logs/dis.log

CUDA_VISIBLE_DEVICES=7 python distill.py \
    --do_eval \
    --train_data_file=../../../data/defect_detection/train.jsonl \
    --eval_data_file=../../../data/defect_detection/test.jsonl \
    --std_model biGRU \
    --vocab_size 10000 \
    --block_size 400 \
    --eval_batch_size 64 \
    --choice recent \
    --seed 123456 2>&1| tee ../logs/dis_eval.log