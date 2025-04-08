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
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);

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
      // Set the most recent call as selected by default
      if (data && data.length > 0) {
        setSelectedCall(data[0]);
      }
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
          <div className="flex justify-between items-center">
            <CardTitle>Conversation Summary</CardTitle>
            {calls.length > 0 && (
              <select
                className="bg-background border rounded-md px-3 py-1 text-sm"
                value={selectedCall?.id || ''}
                onChange={(e) => {
                  const call = calls.find(c => c.id === e.target.value);
                  setSelectedCall(call || null);
                }}
              >
                {calls.map((call) => (
                  <option key={call.id} value={call.id}>
                    {formatDate(call.call_timestamp)} - {formatDuration(call.duration)}
                  </option>
                ))}
              </select>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {selectedCall?.transcript ? (
              <div className="space-y-4">
                {selectedCall.transcript.split('\n').map((line, index) => {
                  if (line.startsWith('AI:')) {
                    return (
                      <div key={index} className="flex justify-start">
                        <div className="max-w-[80%] bg-[#fc583f] rounded-lg p-3">
                          <p className="text-sm font-medium text-white">AI</p>
                          <p className="text-sm text-white">{line.replace('AI:', '').trim()}</p>
                        </div>
                      </div>
                    );
                  } else if (line.startsWith('User:')) {
                    return (
                      <div key={index} className="flex justify-end">
                        <div className="max-w-[80%] bg-accent rounded-lg p-3">
                          <p className="text-sm font-medium text-accent-foreground">User</p>
                          <p className="text-sm">{line.replace('User:', '').trim()}</p>
                        </div>
                      </div>
                    );
                  }
                  return null;
                })}
              </div>
            ) : (
              <p className="text-muted-foreground">No conversation summary available.</p>
            )}
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
              <div 
                key={call.id} 
                className={`flex justify-between items-center p-2 rounded-lg cursor-pointer transition-colors ${
                  selectedCall?.id === call.id ? 'bg-accent' : 'hover:bg-accent/50'
                }`}
                onClick={() => setSelectedCall(call)}
              >
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