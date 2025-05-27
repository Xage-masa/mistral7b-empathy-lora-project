import axios from "axios";

export const sendMessage = async (prompt: string, userId: string) => {
  const response = await axios.post("http://localhost:7860/chat", {
    prompt,
    user_id: userId
  });

  const text = response.data.response;
  return { text, emotion: "neutral" }; // пока заглушка, эмоции добавим позже
};
