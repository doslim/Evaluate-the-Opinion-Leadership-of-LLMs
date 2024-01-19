# python fine-tune.py \
#     --model_version chat-7b \
#     --tokenized_dataset rule_based_qa_baichuan2-7B\
#     --train_size 912 \
#     --eval_size 100 \
#     --lora_rank 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 2 \
#     --save_steps 200 \
#     --evaluation_strategy steps \
#     --eval_steps 25 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --no_prompt_loss 1 \
#     --remove_unused_columns false \
#     --logging_steps 25 \
#     --output_dir weights/rule_based_qa_train_912_baichuan2-7B \
#     --report_to wandb \
#     --run_name rule_based_qa_baichuan2-7B_fine-tuning_0117


python fine-tune.py \
    --model_version chat-13b \
    --tokenized_dataset wwqa_baichuan2-13B\
    --train_size 1453 \
    --eval_size 100 \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --no_prompt_loss 1 \
    --remove_unused_columns false \
    --logging_steps 25 \
    --output_dir weights/wwqa_train_1453_baichuan2-13B \
    --report_to wandb \
    --run_name wwqa_baichuan2-13B_fine-tuning_0117