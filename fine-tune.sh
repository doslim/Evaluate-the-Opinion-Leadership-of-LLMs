
# python fine-tune.py \
#     --model_version Mistral-7B \
#     --tokenized_dataset wwqa_mistral-7B\
#     --train_size 1453 \
#     --eval_size 100 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 4 \
#     --save_steps 25 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --load_best_model_at_end true \
#     --save_total_limit 3 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/wwqa_train_1453_mistral-7B \
#     --report_to wandb \
#     --run_name wwqa_mistral-7B_fine-tuning_0313


# python fine-tune.py \
#     --model_version Baichuan2-13B \
#     --tokenized_dataset wwqa_baichuan2-13B\
#     --train_size 1453 \
#     --eval_size 100 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 42 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 4 \
#     --save_steps 25 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --load_best_model_at_end true \
#     --save_total_limit 3 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/wwqa_train_1453_baichuan2-13B-v2 \
#     --report_to wandb \
#     --run_name wwqa_baichuan2-13B_fine-tuning_0314


# python fine-tune.py \
#     --model_version chatglm3-6b \
#     --tokenized_dataset wwqa_chatglm3-6b\
#     --train_size 1453 \
#     --eval_size 100 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 4\
#     --save_steps 25 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --load_best_model_at_end true \
#     --save_total_limit 3 \
#     --learning_rate 1e-4 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/wwqa_train_1453_chatglm3-6b \
#     --report_to wandb \
#     --run_name wwqa_chatglm3-6b_fine-tuning_0321


python fine-tune.py \
    --model_version internlm2-20b \
    --tokenized_dataset wwqa_internlm2-20b\
    --train_size 1453 \
    --eval_size 100 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4\
    --save_steps 25 \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --load_best_model_at_end true \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --no_prompt_loss 1 \
    --remove_unused_columns false \
    --logging_steps 25 \
    --output_dir weights/wwqa_train_1453_internlm-20b \
    --report_to wandb \
    --run_name wwqa_internlm-20b_fine-tuning_0322