'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { supabase } from '@/lib/supabase';
import type { Caller, Call } from '@/lib/supabase';

// Helper function for consistent date formatting
const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return date.toISOString().split('T')[0]; // Returns YYYY-MM-DD format
};

export default function PatientDetails({ caller }: { caller: Caller }) {
  const [calls, setCalls] = useState<Call[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const fetchCalls = async () => {
      const { data, error } = await supabase
        .from('calls')
        .select('*')
        .eq('phone_number', caller.phone_number)
        .order('call_timestamp', { ascending: false });

      if (error) {
        console.error('Error fetching calls:', error);
        return;
      }

      setCalls(data || []);
    };

    if (caller.phone_number) {
      fetchCalls();

      // Subscribe to changes in the calls table for this caller
      const channel = supabase
        .channel(`calls_${caller.phone_number}`)
        .on('postgres_changes', {
          event: '*',
          schema: 'public',
          table: 'calls',
          filter: `phone_number=eq.${caller.phone_number}`
        }, (payload) => {
          console.log('Call change received!', payload);
          fetchCalls(); // Refetch calls when there's an update
        })
        .subscribe();

      // Cleanup subscription when component unmounts or caller changes
      return () => {
        channel.unsubscribe();
      };
    }
  }, [caller.phone_number]);

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Don't render anything until after client-side hydration
  if (!mounted) {
    return null;
  }

  return (
    <div className="grid gap-6">
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-6">
            <div className="w-32 h-32 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-6xl font-bold">{caller.urgency || '?'}</span>
            </div>
            <div className="space-y-2">
              <h3 className="text-2xl font-bold">{caller.name || 'Anonymous'}</h3>
              <div className="space-y-1">
                <p><span className="font-medium">Phone Number:</span> {caller.phone_number}</p>
                <p><span className="font-medium">Last Contacted:</span> {formatDate(caller.last_call_timestamp)}</p>
                <p><span className="font-medium">Previous History:</span> {caller.previous_history || 'None'}</p>
                <p><span className="font-medium">Sexual Orientation:</span> {caller.sexual_orientation || 'Not specified'}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Conversation Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="whitespace-pre-wrap">
            {caller.aggregated_transcript || 'No conversation summary available.'}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Previous Calls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {calls.map((call) => (
              <div key={call.id} className="flex justify-between items-center">
                <span>{formatDate(call.call_timestamp)}</span>
                <span>Duration: {formatDuration(call.duration)}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 