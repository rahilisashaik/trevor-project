'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { supabase } from '@/lib/supabase';
import type { Caller, Call } from '@/lib/supabase';
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import EmotionRadarChart from './EmotionRadarChart';
import SentimentTimeSeriesChart from './SentimentTimeSeriesChart';

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

  // Add function to filter transcript for key questions
  const filterKeyQuestions = (transcript: string) => {
    const lines = transcript.split('\n');
    const keyQuestions = [
      'How are you feeling?',
      'Have you thought about committing suicide lately?',
      'Do you need urgent help?'
    ];
    
    let filteredLines: string[] = [];
    let foundQuestions = new Set<string>();
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (line.startsWith('AI:')) {
        const aiMessage = line.replace('AI:', '').trim();
        // Check if this is one of our key questions
        const matchingQuestion = keyQuestions.find(q => aiMessage.includes(q));
        if (matchingQuestion && !foundQuestions.has(matchingQuestion)) {
          foundQuestions.add(matchingQuestion);
          filteredLines.push(line);
          // Add the next user response if it exists
          if (i + 1 < lines.length && lines[i + 1].startsWith('User:')) {
            filteredLines.push(lines[i + 1]);
          }
        }
      }
    }
    
    return filteredLines.join('\n');
  };

  // Don't render anything until after client-side hydration
  if (!mounted) {
    return <div className="grid gap-6">
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-6">
            <div className="w-32 h-32 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-6xl font-bold">...</span>
            </div>
            <div className="space-y-2">
              <h3 className="text-2xl font-bold">Loading...</h3>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>;
  }

  return (
    <div className="grid gap-6">
      <div className="grid grid-cols-3 gap-6">
        <Card className="col-span-2">
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
            <CardDescription>
              {selectedCall ? (
                <span>Summary from {formatDate(selectedCall.call_timestamp)}</span>
              ) : (
                'Select a call to view the summary'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedCall?.summary ? (
              <div className="prose prose-sm max-w-none">
                {selectedCall.summary.split('\n').map((paragraph, index) => (
                  <p key={index} className="text-muted-foreground">
                    {paragraph}
                  </p>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground">No summary available for this call</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Emotion Analysis</CardTitle>
            <CardDescription>
              {selectedCall ? (
                <span>Emotional state from {formatDate(selectedCall.call_timestamp)}</span>
              ) : (
                'Select a call to view emotion analysis'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedCall?.emotion_scores ? (
              <EmotionRadarChart emotionScores={selectedCall.emotion_scores} />
            ) : (
              <p className="text-muted-foreground">No emotion analysis available for this call</p>
            )}
          </CardContent>
        </Card>

        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>Sentiment Analysis Over Time</CardTitle>
            <CardDescription>
              {selectedCall ? (
                <span>Sentiment progression from {formatDate(selectedCall.call_timestamp)}</span>
              ) : (
                'Select a call to view sentiment analysis'
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedCall?.sentiment_time_series ? (
              <SentimentTimeSeriesChart sentimentData={selectedCall.sentiment_time_series} />
            ) : (
              <p className="text-muted-foreground">No sentiment time series data available for this call</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Key Questions & Responses</CardTitle>
          <CardDescription>
            {selectedCall ? (
              <div className="flex items-center justify-between">
                <span>Call from {formatDate(selectedCall.call_timestamp)}</span>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      Select Call
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {calls.map((call) => (
                      <DropdownMenuItem
                        key={call.id}
                        onClick={() => setSelectedCall(call)}
                        className={selectedCall?.id === call.id ? 'bg-accent' : ''}
                      >
                        {formatDate(call.call_timestamp)}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ) : (
              'Select a call to view the responses'
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {selectedCall?.transcript ? (
            <div className="space-y-4">
              {filterKeyQuestions(selectedCall.transcript).split('\n').map((line, index) => {
                if (line.startsWith('AI:')) {
                  return (
                    <div key={index} className="flex flex-col items-start">
                      <span className="text-sm text-white mb-1">AI</span>
                      <div className="bg-[#fc583f] text-white p-3 rounded-lg max-w-[80%]">
                        {line.replace('AI:', '').trim()}
                      </div>
                    </div>
                  );
                } else if (line.startsWith('User:')) {
                  return (
                    <div key={index} className="flex flex-col items-end">
                      <span className="text-sm text-muted-foreground mb-1">{caller.name || 'Anonymous'}</span>
                      <div className="bg-gray-100 text-gray-900 p-3 rounded-lg max-w-[80%]">
                        {line.replace('User:', '').trim()}
                      </div>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          ) : (
            <p className="text-muted-foreground">No transcript available for this call</p>
          )}
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