import React from "react";

const Avatar = ({ emotion }: { emotion: string }) => {
  return (
    <div className="mt-4 text-center text-sm text-gray-500">
      Персонаж: настроение — {emotion}
    </div>
  );
};

export default Avatar;
