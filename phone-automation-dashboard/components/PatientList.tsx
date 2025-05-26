'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import type { Caller } from '@/lib/supabase';
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ChevronDown, ChevronUp } from "lucide-react";

type SortField = 'urgency' | 'lastCall';
type SortOrder = 'asc' | 'desc';

export default function PatientList({ onSelect }: { onSelect: (caller: Caller) => void }) {
  const [callers, setCallers] = useState<(Caller & { urgency_score?: number })[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [sortField, setSortField] = useState<SortField>('lastCall');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');

  const getUrgencyColor = (score: number | undefined) => {
    if (!score) return 'bg-gray-400';
    if (score >= 8) return 'bg-red-500';
    if (score >= 5) return 'bg-orange-500';
    if (score >= 3) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const sortCallers = (callers: (Caller & { urgency_score?: number })[], field: SortField, order: SortOrder) => {
    return [...callers].sort((a, b) => {
      if (field === 'urgency') {
        const scoreA = a.urgency_score || 0;
        const scoreB = b.urgency_score || 0;
        return order === 'asc' ? scoreA - scoreB : scoreB - scoreA;
      } else {
        const dateA = new Date(a.last_call_timestamp).getTime();
        const dateB = new Date(b.last_call_timestamp).getTime();
        return order === 'asc' ? dateA - dateB : dateB - dateA;
      }
    });
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Toggle order if clicking the same field
      setSortOrder(order => order === 'asc' ? 'desc' : 'asc');
    } else {
      // Default to descending order when changing fields
      setSortField(field);
      setSortOrder('desc');
    }
  };

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const fetchCallersAndScores = async () => {
      const { data: callersData, error: callersError } = await supabase
        .from('callers')
        .select('*')
        .order('last_call_timestamp', { ascending: false });

      if (callersError) {
        console.error('Error fetching callers:', callersError);
        return;
      }

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

      // Sort the data based on current sort field and order
      const sortedCallers = sortCallers(callersWithScoresData, sortField, sortOrder);
      setCallers(sortedCallers);

      if (selectedId) {
        const selectedCaller = sortedCallers.find(c => c.phone_number === selectedId);
        if (selectedCaller) {
          onSelect(selectedCaller);
        }
      }
    };

    if (mounted) {
      fetchCallersAndScores();

      const callersChannel = supabase
        .channel('callers_changes')
        .on('postgres_changes', { event: '*', schema: 'public', table: 'callers' }, fetchCallersAndScores)
        .subscribe();

      const callsChannel = supabase
        .channel('calls_changes')
        .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'calls' }, fetchCallersAndScores)
        .subscribe();

      return () => {
        callersChannel.unsubscribe();
        callsChannel.unsubscribe();
      };
    }
  }, [selectedId, onSelect, mounted, sortField, sortOrder]);

  // Loading state
  if (!mounted) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="p-3 rounded-lg animate-pulse bg-accent/20">
            <div className="flex justify-between items-center">
              <span className="font-medium">Loading...</span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between px-2">
        <span className="text-sm font-medium">Sort by:</span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSort('urgency')}
            className="flex items-center gap-1"
          >
            Urgency
            {sortField === 'urgency' && (
              sortOrder === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSort('lastCall')}
            className="flex items-center gap-1"
          >
            Last Call
            {sortField === 'lastCall' && (
              sortOrder === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

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
                Last Interaction: {new Date(caller.last_call_timestamp).toLocaleDateString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 