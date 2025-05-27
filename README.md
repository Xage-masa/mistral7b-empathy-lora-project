# Emotional AI Companion (Mistral-7B + LoRA)

Дипломный проект: ИИ-компаньон, способный оказывать первичную эмоциональную поддержку пользователям в состоянии тревоги, одиночества или неуверенности.

Проект реализован с использованием модели Mistral-7B, дообученной через LoRA на эмпатичном корпусе. Интерфейс представлен в виде анимированного персонажа, реагирующего на эмоциональный контекст диалога.

## Основные технологии

- Mistral-7B (через Hugging Face)
- LoRA дообучение (PEFT)
- FastAPI (сервер для генерации ответов)
- React + Tailwind (фронтенд)
- Анимированный персонаж (реакции: idle, speak, worry, proud и др.)
- Контейнеризация через Docker
- Визуализация обучения в Weights & Biases

---

<p align="center">
  <img src="https://github.com/Xage-masa/mistral7b-empathy-lora-project/blob/main/frontend/public/gif/idle.gif?raw=true" width="220" alt="AI-компаньон приветствует вас">
</p>

<p align="center"><em>ИИ-компаньон Широ приветствует вас и готова к работе</em></p>

---



```markdown
## Структура проекта

```text
 .
 ├── frontend/                # Интерфейс пользователя
 │   └── public/character/    # Анимации персонажа
 ├── server/                  # API + модель
 │   └── bot/                 # Генерация и логика ответов
 ├── .gitignore
 └── package.json             # Корневые зависимости (Docker/Dev)



````
<p align="center">
  <img src="https://github.com/Xage-masa/mistral7b-empathy-lora-project/blob/main/frontend/public/gif/read.gif?raw=true" width="300" alt="reading">
</p>

<p align="center"><em>Пока все устанавливается, можешь немного почитать о проекте… Или просто поболтать с Широ. Она всегда рядом.</em></p>

## Быстрый запуск

### Фронтенд

```bash
cd frontend
npm install
npm run dev
````

### Сервер (FastAPI)

```bash
cd server
uvicorn bot.main:app --reload
```

> Зависимости указаны в `server/requirements.txt` 
---


### Мониторинг обучения

Графики обучения, веса, градиенты и ресурсы GPU зафиксированы в Weights & Biases:  
[Открыть W&B репорт](https://api.wandb.ai/links/summonerlin-geekbrains/aip54d0n)

<p align="center">
  <img src="https://github.com/Xage-masa/mistral7b-empathy-lora-project/blob/main/frontend/public/gif/worry2.gif?raw=true" width="220" alt="AI-компаньон тревожится">
</p>

<p align="center"><em>Широ слегка тревожится за стабильность градиентов</em></p>



---

### Автор  
**Diana Grodik**  
**Год:** 2025  
Проект реализован в рамках дипломной работы на тему:  
*«ИИ-компаньон как инструмент первичной эмоциональной поддержки»*
