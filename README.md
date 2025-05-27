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
  <img src="frontend/public/character/gif/idle.gif" width="220" alt="AI-компаньон приветствует вас">
</p>

<p align="center"><em>ИИ-компаньон приветствует вас и готов к работе</em></p>

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
  <img src="frontend/public/character/gif/worry.gif" width="220" alt="AI-компаньон переживает за обучение">
</p>

<p align="center"><em>ИИ-компаньон слегка тревожится за стабильность градиентов</em></p>


---

### Автор  
**Diana Grodik**  
**Год:** 2025  
Проект реализован в рамках дипломной работы на тему:  
*«ИИ-компаньон как инструмент первичной эмоциональной поддержки»*
