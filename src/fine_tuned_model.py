from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, EarlyStoppingCallback
import torch
import tokenize_datasets as td
import os

#clear cache
torch.cuda.empty_cache()

#check if cuda is available or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

#training argument define
training_args = TrainingArguments(
    output_dir="/home/nil/python_projects/gpt2_45k_minor_project/results",
    eval_strategy="steps",  # Evaluate every few steps
    save_steps=2500,       
    eval_steps=2500,         
    num_train_epochs=12,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10000,
    weight_decay=0.01,
    logging_dir="/home/nil/python_projects/gpt2_45k_minor_project/logs",
    logging_strategy="steps",  # Log every 'logging_steps'
    logging_steps=50,  # Log every 50 steps
    learning_rate=3e-5,
    report_to=["tensorboard"],   
    fp16=True,  # Use mixed precision training
    save_total_limit=8,  
    load_best_model_at_end=True, 
    metric_for_best_model="loss",  # Use loss to decide best model
    greater_is_better=False,  # Lower loss is better
    log_level="info"
)

# Early Stopping Callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

#Trainer initialization
trainer = Trainer(
    model =model,
    args = training_args,
    train_dataset= td.final_dataset['train'],
    eval_dataset=td.final_dataset['test'],
    tokenizer = td.tokenizer,
    callbacks= [early_stopping]
)

if __name__ == "__main__":
    # checkpoint_path = '/home/nil/python_projects/gpt2_45k_minor_project/results'
    # if os.path.exists(checkpoint_path):
    #     trainer.train(resume_from_checkpoint=checkpoint_path)
    # else:
    #     trainer.train()

    trainer.train()

    # Save model
    model_output_dir = '/home/nil/python_projects/gpt2_45k_minor_project/results/final_model'
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir)
    td.tokenizer.save_pretrained(model_output_dir)