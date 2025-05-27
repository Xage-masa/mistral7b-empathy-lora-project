import { useState } from "react";
import CompanionAnimation from "./components/CompanionAnimation";

// Тип состояний 
type Mood =
  | "idle"
  | "speak"
  | "worry"
  | "proud"
  | "games"
  | "hello"
  | "breathe"
  | "read"
  | "meditation";

export default function App() {
  const [mood, setMood] = useState<Mood>("idle");

  const handleUserMessage = async (message: string) => {
    // Анализ пользовательского ввода
    if (message.includes("тревога")) setMood("worry");
    else if (message.includes("игра")) setMood("games");
    else if (message.includes("читать")) setMood("read");
    else setMood("breathe");

    // Отправка на сервер 
    const response = await sendMessageToBot(message);

    // Анализ ответа от модели
    if (response.includes("горжусь")) setMood("proud");
    else if (response.includes("спокойно")) setMood("meditation");
    else if (response.includes("привет")) setMood("hello");
    else setMood("speak");

    // Возврат к нейтральному состоянию через 4 секунды
    setTimeout(() => setMood("idle"), 4000);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <CompanionAnimation mood={mood} />

      {/* поле и кнопка */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const input = e.currentTarget.elements.namedItem("message") as HTMLInputElement;
          const msg = input.value.trim();
          if (msg) handleUserMessage(msg);
          input.value = "";
        }}
        className="mt-4 flex gap-2"
      >
        <input
          name="message"
          type="text"
          placeholder="Напиши что-нибудь..."
          className="border px-4 py-2 rounded-lg w-80"
        />
        <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded-lg">
          Отправить
        </button>
      </form>
    </div>
  );
}


async function sendMessageToBot(prompt: string): Promise<string> {
  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  });

  const data = await response.json();
  return data.response || "";
}
