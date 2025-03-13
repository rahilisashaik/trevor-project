"use client";
import { useState, useEffect } from "react";
import Preloader from "@/components/Preloader";

export default function LoadingWrapper({ children }: { children: React.ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    setTimeout(() => {
      setLoading(false);
      setTimeout(() => setShowContent(true), 100); // Delay before fade-in
    }, 3000); // Simulated loading time
  }, []);

  return (
    <>
      {loading ? (
        <Preloader onComplete={() => setLoading(false)} />
      ) : (
        <div
          className={`opacity-0 transition-opacity duration-1000 ${showContent ? "opacity-100" : ""}`}
        >
          {children}
        </div>
      )}
    </>
  );
}
