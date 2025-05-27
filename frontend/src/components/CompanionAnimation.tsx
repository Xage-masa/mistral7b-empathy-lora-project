import React from "react";

type Mood =
  | "idle"
  | "speak"
  | "breathe"
  | "games"
  | "hello"
  | "meditation"
  | "proud"
  | "read"
  | "worry";

interface Props {
  mood: Mood;
}

export default function CompanionAnimation({ mood }: Props) {
  const src = `/character/${mood}.mp4`;

  return (
    <div className="flex justify-center items-center p-4">
      <video
        key={mood} // позволяет перезапустить при смене
        src={src}
        autoPlay
        loop
        muted
        playsInline
        className="w-64 h-auto rounded-2xl shadow-lg"
      />
    </div>
  );
}
