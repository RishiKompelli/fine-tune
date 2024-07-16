import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
from sklearn.model_selection import train_test_split
import re
import csv
import numpy as np
import multiprocessing as mp
from datasets import load_dataset

dataset = load_dataset("RishiKompelli/TherapyDataset")

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Fix newline characters in the DataFrame
def fix_newlines(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.sub(r'\n', ' ', x) if isinstance(x, str) else x)
    return df

df = fix_newlines(df, ['input', 'output'])

if 'input' not in df.columns or 'output' not in df.columns:
    raise ValueError("Columns 'input' and/or 'output' not found in the cleaned dataset.")

inputs = df['input'].astype(str).tolist()
outputs = df['output'].astype(str).tolist()

# Load dataset
dataset = load_dataset("RishiKompelli/TherapyDataset")

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Fix newline characters in the DataFrame
def fix_newlines(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.sub(r'\n', ' ', x) if isinstance(x, str) else x)
    return df

df = fix_newlines(df, ['input', 'output'])

if 'input' not in df.columns or 'output' not in df.columns:
    raise ValueError("Columns 'input' and/or 'output' not found in the cleaned dataset.")

inputs = df['input'].astype(str).tolist()
outputs = df['output'].astype(str).tolist()

# Tokenize the data using GPT-2 tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have an official padding token, so we use the EOS token

def tokenize_texts(inputs, outputs, tokenizer):
    input_output_pairs = [f"{inp} {tokenizer.eos_token} {out}" for inp, out in zip(inputs, outputs)]
    return tokenizer(input_output_pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')

def parallel_tokenization(inputs, outputs, tokenizer):
    chunk_size = len(inputs) // mp.cpu_count()
    input_chunks = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]
    output_chunks = [outputs[i:i + chunk_size] for i in range(0, len(outputs), chunk_size)]

    with mp.Pool(mp.cpu_count()) as pool:
        encodings_list = pool.starmap(tokenize_texts, [(input_chunk, output_chunk, tokenizer) for input_chunk, output_chunk in zip(input_chunks, output_chunks)])

    encodings = {key: torch.cat([enc[key] for enc in encodings_list], dim=0) for key in encodings_list[0].keys()}
    return encodings

encodings = parallel_tokenization(inputs, outputs, tokenizer)

# Create Custom Dataset
class CustomLMHeadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]  # For GPT models, labels are typically the same as input_ids
        }

    def __len__(self):
        return len(self.input_ids)

# Split the dataset
train_size = int(0.9 * len(encodings['input_ids']))
val_size = len(encodings['input_ids']) - train_size

train_dataset = CustomLMHeadDataset({k: v[:train_size] for k, v in encodings.items()})
val_dataset = CustomLMHeadDataset({k: v[train_size:] for k, v in encodings.items()})

# Set Up Data Collator and Model
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masking for GPT models
)

model = GPT2LMHeadModel.from_pretrained(model_name)

# Configure training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    save_steps=1000,
    save_total_limit=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Add error handling
try:
    trainer.train()
    print("Fine-tuning completed successfully")
    model.save_pretrained('./finetuned_gpt2')
    tokenizer.save_pretrained('./finetuned_gpt2')
except Exception as e:
    print(f"An error occurred during fine-tuning: {str(e)}")