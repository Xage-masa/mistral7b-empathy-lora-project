import json
from datetime import datetime

# === Настройки
synthetic_files = [
    "synthetic_dataset_20250510_232428.jsonl",
    "followup_dataset_20250510_235647.jsonl"
]

multi_turn_files = [
    "1multi_turn_dataset_20250511_004659.jsonl",
    "2multi_turn_dataset_20250511_012214.jsonl",
    "3multi_turn_dataset_20250511_012954.jsonl"
]

output_synthetic = f"combined_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
output_multiturn = f"combined_multiturn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# === Объединяем обычные (prompt-response)
with open(output_synthetic, "w", encoding="utf-8") as fout:
    for fname in synthetic_files:
        with open(fname, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line if line.endswith("\n") else line + "\n")
print(f" Сохранено: {output_synthetic}")

# === Объединяем multi-turn диалоги
with open(output_multiturn, "w", encoding="utf-8") as fout:
    for fname in multi_turn_files:
        with open(fname, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line if line.endswith("\n") else line + "\n")
print(f" Сохранено: {output_multiturn}")
