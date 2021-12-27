mkdir logs
mkdir small

CUDA_VISIBLE_DEVICES=0,1 python main.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../../data/code_repair/small/train.buggy-fixed.buggy,../../data/code_repair/small/train.buggy-fixed.fixed \
	--dev_filename ../../data/code_repair/small/valid.buggy-fixed.buggy,../../data/code_repair/small/valid.buggy-fixed.fixed \
	--output_dir ./small \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path ./small/checkpoint-best-bleu/model.bin \
	--dev_filename ../../data/code_repair/small/valid.buggy-fixed.buggy,../../data/code_repair/small/valid.buggy-fixed.fixed \
	--test_filename ../../data/code_repair/small/test.buggy-fixed.buggy,../../data/code_repair/small/test.buggy-fixed.fixed \
	--output_dir ./small \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--eval_batch_size 16 