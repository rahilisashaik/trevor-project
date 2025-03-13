"use client";
import { useState, useEffect } from "react";
import { Progress } from "@/components/ui/progress";

export default function Preloader({ onComplete }: { onComplete: () => void }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((oldProgress) => {
        if (oldProgress >= 100) {
          clearInterval(interval);
          setTimeout(() => onComplete(), 500); // Small delay before hiding
          return 100;
        }
        return oldProgress + (100 - oldProgress) * 0.1; // Easing effect
      });
    }, 50); // Faster update interval

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-background">
      <div className="w-1/2">
        <Progress value={progress} />
      </div>
    </div>
  );
}
