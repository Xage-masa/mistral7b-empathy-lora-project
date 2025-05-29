from openai import OpenAI
import json
import os
import sys
import time

# Настраиваем консольный вывод в UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# API-ключ OpenAI
client = OpenAI api_key=

# Пути к файлам
input_path = "empathy_all.txt"
output_path = "empathy_dataset_ru.jsonl"

# Максимум новых строк за запуск
MAX_LINES = 2000

# Сколько уже сгенерировано
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as fout:
        processed_lines = fout.readlines()
        done = len(processed_lines)
else:
    done = 0

# Загружаем весь датасет
with open(input_path, "r", encoding="utf-8") as fin:
    lines = [line.strip() for line in fin if line.strip()]

# Определяем, сколько обрабатывать
lines_to_process = lines[done : done + MAX_LINES]

print(f"Всего строк в исходном датасете: {len(lines)}")
print(f"Уже обработано: {done}")
print(f"Осталось обработать: {len(lines_to_process)}\n")

# Промт-система
system_prompt = "Ты — эмпатичный ИИ-друг. Отвечай мягко, по-доброму, дружелюбно, но не навязчиво. Сохраняй человечность и легкость."
"Ты — эмпатичный ИИ-друг. Твоя задача — поддерживать собеседника, говорить с ним дружелюбно и с пониманием, словно ты близкий человек, готовый выслушать и утешить."

# Генерация и сохранение
for i, line in enumerate(lines_to_process, start=done + 1):
    print(f"[{i}] Генерирую ответ для: {line}")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": line}
            ],
            temperature=0.85,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f" Ошибка генерации: {e}")
        continue

    try:
        with open(output_path, "a", encoding="utf-8") as fout:
            json.dump({
                "instruction": "Ответь как добрый ИИ-друг",
                "input": line,
                "output": reply
            }, fout, ensure_ascii=False)
            fout.write("\n")
    except Exception as e:
        print(f" Ошибка сохранения: {e}")

    time.sleep(1.5)  # Уважение к лимитам

print("\n Завершено.")