'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import type { Caller } from '@/lib/supabase';

export default function PatientList({ onSelect }: { onSelect: (caller: Caller) => void }) {
  const [callers, setCallers] = useState<Caller[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchCallers = async () => {
      const { data, error } = await supabase
        .from('callers')
        .select('*')
        .order('last_call_timestamp', { ascending: false });

      if (error) {
        console.error('Error fetching callers:', error);
        return;
      }

      setCallers(data || []);

      // If we have a selected caller, update their data
      if (selectedId) {
        const selectedCaller = data?.find(c => c.phone_number === selectedId);
        if (selectedCaller) {
          onSelect(selectedCaller);
        }
      }
    };

    fetchCallers();

    // Subscribe to all changes in the callers table
    const channel = supabase
      .channel('callers_changes')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'callers'
      }, (payload) => {
        console.log('Caller change received!', payload);
        fetchCallers(); // Refetch callers when there's an update
      })
      .subscribe();

    // Cleanup subscription
    return () => {
      channel.unsubscribe();
    };
  }, [selectedId, onSelect]);

  return (
    <div className="space-y-2">
      {callers.map((caller) => (
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
          <div className="flex justify-between items-center">
            <span className="font-medium">{caller.name || 'Anonymous'}</span>
            <span className="text-sm text-muted-foreground">
              Urgency: {caller.urgency || 'N/A'}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
} 