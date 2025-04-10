'use client';

import { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

interface EmotionScores {
  happy: number;
  fearful: number;
  sad: number;
  surprised: number;
  neutral: number;
  angry: number;
  disgust: number;
}

interface EmotionRadarChartProps {
  emotionScores: EmotionScores | null;
}

export default function EmotionRadarChart({ emotionScores }: EmotionRadarChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!chartRef.current || !emotionScores) return;

    // Destroy existing chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    const emotions = Object.keys(emotionScores);
    const scores = Object.values(emotionScores);

    chartInstance.current = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
        datasets: [{
          label: 'Emotion Intensity',
          data: scores,
          backgroundColor: 'rgba(252, 88, 63, 0.2)',
          borderColor: 'rgba(252, 88, 63, 1)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(252, 88, 63, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(252, 88, 63, 1)'
        }]
      },
      options: {
        scales: {
          r: {
            beginAtZero: true,
            min: 0,
            max: 1,
            ticks: {
              stepSize: 0.2
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [emotionScores]);

  if (!emotionScores) {
    return <div className="text-center text-muted-foreground">No emotion data available</div>;
  }

  return (
    <div className="w-full h-[300px] relative">
      <canvas ref={chartRef} />
    </div>
  );
} 