import React, { useState } from "react";
import MessageBubble from "./MessageBubble";
import Avatar from "./Avatar";

const ChatWindow = () => {
  const [messages, setMessages] = useState([{ from: "bot", text: "Привет! Я твой ИИ-друг." }]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
  if (!input.trim()) return;

  setMessages([...messages, { from: "user", text: input }]);
  setIsTyping(true);

  const { text, emotion } = await fetchFromBackend(input, "user-001");

  setIsTyping(false);
  setMessages((prev) => [...prev, { from: "bot", text, emotion }]);
  setInput("");
};

  return (
    <div className="bg-white shadow-xl p-4 rounded-lg max-w-lg w-full">
      <div className="space-y-2 mb-4">
        {messages.map((m, i) => (
          <MessageBubble key={i} from={m.from} text={m.text} />
        ))}
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="border rounded px-3 py-2 flex-1"
          placeholder="Напиши что-нибудь..."
        />
        <button onClick={sendMessage} className="bg-blue-500 text-white px-4 py-2 rounded">
          Отправить
        </button>
      </div>
      <Avatar emotion="neutral" />
    </div>
  );
};

export default ChatWindow;
