import React from "react";

const MessageBubble = ({ from, text }: { from: string; text: string }) => {
  const isUser = from === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`px-4 py-2 rounded-lg ${isUser ? "bg-blue-100" : "bg-gray-200"}`}>
        <span>{text}</span>
      </div>
    </div>
  );
};

export default MessageBubble;
