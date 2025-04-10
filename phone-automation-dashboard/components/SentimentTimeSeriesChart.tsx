'use client';

import { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

interface SentimentDataPoint {
  timestamp: number;
  score: number;
  rms: number;
  zcr: number;
}

interface SentimentTimeSeriesChartProps {
  sentimentData: SentimentDataPoint[] | null;
}

export default function SentimentTimeSeriesChart({ sentimentData }: SentimentTimeSeriesChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!chartRef.current || !sentimentData) return;

    // Destroy existing chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Prepare data
    const timestamps = sentimentData.map(point => point.timestamp);
    const scores = sentimentData.map(point => point.score);

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: timestamps.map(t => `${t}s`),
        datasets: [{
          label: 'Sentiment Score',
          data: scores,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
          fill: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            min: -1,
            max: 1,
            title: {
              display: true,
              text: 'Sentiment Score'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time (seconds)'
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                const point = sentimentData[context.dataIndex];
                return [
                  `Score: ${point.score.toFixed(2)}`,
                  `RMS: ${point.rms.toFixed(2)}`,
                  `ZCR: ${point.zcr.toFixed(2)}`
                ];
              }
            }
          }
        }
      }
    });

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [sentimentData]);

  if (!sentimentData) {
    return <div className="text-center text-muted-foreground">No sentiment data available</div>;
  }

  return (
    <div className="w-full h-[300px] relative">
      <canvas ref={chartRef} />
    </div>
  );
}