mkdir -p logs/cs

CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --do_train \
    --train_data_file=../../data/code_translation/train.java-cs.txt.cs,../../data/code_translation/train.java-cs.txt.cs \
    --eval_data_file=../../data/code_translation/valid.java-cs.txt.cs,../../data/code_translation/valid.java-cs.txt.cs \
    --output_dir ./cs \
    --epoch 40 \
    --beam_size 5 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee logs/cs/train.log

CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --do_eval \
    --eval_data_file=../../data/code_translation/test.java-cs.txt.cs,../../data/code_translation/test.java-cs.txt.cs \
    --output_dir ./cs \
    --block_size 256 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee logs/cs/eval.log