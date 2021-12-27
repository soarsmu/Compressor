# mkdir java

# CUDA_VISIBLE_DEVICES=2,3 python main.py \
# 	--do_train \
# 	--do_eval \
# 	--model_type roberta \
# 	--model_name_or_path microsoft/codebert-base \
# 	--config_name roberta-base \
# 	--tokenizer_name roberta-base \
# 	--train_filename ../../data/code_translation/train.java-cs.txt.java,../../data/code_translation/train.java-cs.txt.cs \
# 	--dev_filename ../../data/code_translation/valid.java-cs.txt.java,../../data/code_translation/valid.java-cs.txt.cs \
# 	--output_dir ./java \
# 	--max_source_length 430 \
# 	--max_target_length 430 \
# 	--beam_size 5 \
# 	--train_batch_size 16 \
# 	--eval_batch_size 16 \
# 	--learning_rate 5e-5 \
# 	--train_steps 100000 \
# 	--eval_steps 5000

CUDA_VISIBLE_DEVICES=6,7 python main.py \
    --do_test \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path ./java/checkpoint-best-bleu/model.bin \
	--dev_filename ../../data/code_translation/valid.java-cs.txt.java,../../data/code_translation/valid.java-cs.txt.cs \
	--test_filename ../../data/code_translation/test.java-cs.txt.java,../../data/code_translation/test.java-cs.txt.cs \
	--output_dir ./java \
	--max_source_length 430 \
	--max_target_length 430 \
	--beam_size 5 \
	--eval_batch_size 16 