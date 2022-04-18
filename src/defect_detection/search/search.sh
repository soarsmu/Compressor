CUDA_VISIBLE_DEVICES=0 python search.py \
    --population_size 10 \
    --generation_size 2 \
    --train_data_file=../../../data/defect_detection/search_train.jsonl \
    --eval_data_file=../../../data/defect_detection/valid.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee logs/search.log