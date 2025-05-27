from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

app = FastAPI()

# === Настройки ===
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = os.environ.get("ADAPTER_PATH", "./model")
hf_token = os.environ.get("HF_TOKEN", None)

print("=" * 40)
print(" Загружаем токенайзер...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=hf_token,
    use_fast=False,
    force_download=True
)


print(" Загружаем базовую модель...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=hf_token,
    trust_remote_code=True
)

print(f" Загружаем адаптер LoRA из: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print(" Адаптер подключен. Модель готова.")

# === История сообщений для каждого пользователя ===
user_histories = {}

def format_chat(history):
    messages = []
    for role, content in history:
        messages.append({"role": role, "content": content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_id = body.get("user_id", "default")
    prompt = body.get("prompt", "").strip()

    if not prompt:
        return {"response": "Ошибка: пустой запрос."}

    # Инициализируем историю, если нет
    if user_id not in user_histories:
        user_histories[user_id] = []

    # Добавляем пользовательский ввод
    user_histories[user_id].append(("user", prompt))

    # Генерируем ответ
    chat_prompt = format_chat(user_histories[user_id])
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text[len(chat_prompt):].strip()

    # Добавляем ответ бота в историю
    user_histories[user_id].append(("assistant", answer))

    return {"response": answer}
