CUDA_VISIBLE_DEVICES=7 python distill.py \
    --do_eval \
    --train_data_file=../../data/defect_detection/train.jsonl \
    --eval_data_file=../../data/defect_detection/valid.jsonl \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee logs/dis.log