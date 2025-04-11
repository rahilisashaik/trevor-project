'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import type { Caller, Call } from '@/lib/supabase';

export default function PatientList({ onSelect }: { onSelect: (caller: Caller) => void }) {
  const [callers, setCallers] = useState<Caller[]>([]);
  const [callersWithScores, setCallersWithScores] = useState<(Caller & { urgency_score?: number })[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  const getUrgencyColor = (score: number | undefined) => {
    if (!score) return 'bg-gray-400';
    if (score >= 8) return 'bg-red-500';
    if (score >= 5) return 'bg-orange-500';
    if (score >= 3) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const fetchCallersAndScores = async () => {
      // First fetch all callers
      const { data: callersData, error: callersError } = await supabase
        .from('callers')
        .select('*')
        .order('last_call_timestamp', { ascending: false });

      if (callersError) {
        console.error('Error fetching callers:', callersError);
        return;
      }

      setCallers(callersData || []);

      // For each caller, fetch their most recent call to get the urgency score
      const callersWithScoresData = await Promise.all((callersData || []).map(async (caller) => {
        const { data: callsData } = await supabase
          .from('calls')
          .select('urgency_score')
          .eq('phone_number', caller.phone_number)
          .order('call_timestamp', { ascending: false })
          .limit(1);

        return {
          ...caller,
          urgency_score: callsData?.[0]?.urgency_score
        };
      }));

      setCallersWithScores(callersWithScoresData);

      // If we have a selected caller, update their data
      if (selectedId) {
        const selectedCaller = callersWithScoresData.find(c => c.phone_number === selectedId);
        if (selectedCaller) {
          onSelect(selectedCaller);
        }
      }
    };

    if (mounted) {
      fetchCallersAndScores();

      // Subscribe to changes in both callers and calls tables
      const callersChannel = supabase
        .channel('callers_changes')
        .on('postgres_changes', {
          event: '*',
          schema: 'public',
          table: 'callers'
        }, () => {
          fetchCallersAndScores();
        })
        .subscribe();

      const callsChannel = supabase
        .channel('calls_changes')
        .on('postgres_changes', {
          event: 'INSERT',
          schema: 'public',
          table: 'calls'
        }, () => {
          fetchCallersAndScores();
        })
        .subscribe();

      // Cleanup subscriptions
      return () => {
        callersChannel.unsubscribe();
        callsChannel.unsubscribe();
      };
    }
  }, [selectedId, onSelect, mounted]);

  if (!mounted) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="p-3 rounded-lg animate-pulse bg-accent/20"
          >
            <div className="flex justify-between items-center">
              <span className="font-medium">Loading...</span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {callersWithScores.map((caller) => (
        <div
          key={caller.phone_number}
          className={`p-3 rounded-lg cursor-pointer transition-colors ${
            selectedId === caller.phone_number ? 'bg-accent' : 'hover:bg-accent/50'
          }`}
          onClick={() => {
            setSelectedId(caller.phone_number);
            onSelect(caller);
          }}
        >
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <span className="font-medium">{caller.name || 'Anonymous'}</span>
              <div className={`flex items-center gap-2 rounded-full px-2 py-1 text-white ${getUrgencyColor(caller.urgency_score)}`}>
                <span className="text-xs font-medium">
                  {typeof caller.urgency_score === 'number' ? caller.urgency_score : 'No Score'}
                </span>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              Last Call: {new Date(caller.last_call_timestamp).toLocaleDateString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
} 