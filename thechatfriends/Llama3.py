import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset

# 모델과 토크나이저 로드
model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 데이터셋 로드
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

dataset = load_dataset("./friends_conversation_dataset_0617.csv")

# 데이터셋 토큰화
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# QLoRA 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 모델 훈련
trainer.train()
