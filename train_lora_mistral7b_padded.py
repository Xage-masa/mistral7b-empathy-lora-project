import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import wandb

# === Загрузка переменных окружения ===
load_dotenv("wandb.env")

# === Авторизация ===
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    hf_login(token=hf_token)

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
    wandb.login(key=wandb_api_key)

# === Загрузка конфигурации ===
with open("config_mistral7b.json", "r") as f:
    cfg = json.load(f)

# === WandB init ===
if cfg.get("use_wandb", False):
    wandb.init(project=cfg["wandb_project"], config=cfg)

# === Модель и токенизатор ===
model_name = cfg["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ВАЖНО: задаем pad_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# === LoRA ===
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# === Проверка trainable параметров ===
model.print_trainable_parameters()

# === Обработка датасета ===
def format_row(example):
    text = f"Пользователь: {example['input']}\nИИ: {example['output']}"
    ids = tokenizer(text, truncation=True, padding="max_length", max_length=512)["input_ids"]
    return {
        "input_ids": ids,
        "labels": ids
    }

dataset = load_dataset("json", data_files=cfg["dataset_path"])["train"]
dataset = dataset.map(format_row, remove_columns=dataset.column_names)

# === Аргументы тренировки ===
args = TrainingArguments(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["batch_size"],
    num_train_epochs=cfg["epochs"],
    learning_rate=cfg["lr"],
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb" if cfg.get("use_wandb") else "none",
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# === Обучение ===
trainer.train()

# === Сохранение модели ===
trainer.save_model(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])