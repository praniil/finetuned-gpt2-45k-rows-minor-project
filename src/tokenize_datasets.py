import datasets_loading_preprocessing as dlp
from transformers import AutoTokenizer
from datasets import concatenate_datasets

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_dataset(examples):
    text = [f"<|input|> {inp} <|output|> {out}" for inp, out in zip(examples['input'], examples['output'])]
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_dataset = [ds.map(tokenize_dataset, batched=True) for ds in dlp.relavant_columns_datasets]

#Concatenation of dataset
combined_dataset = concatenate_datasets([ds['train'] for ds in tokenized_dataset if 'train' in ds])
train_val_test = combined_dataset.train_test_split(test_size=0.2, seed = 42)

final_dataset = {
    'train': train_val_test['train'],
    'test': train_val_test['test']
}

print(len(final_dataset['train']))
print(len(final_dataset['test']))
