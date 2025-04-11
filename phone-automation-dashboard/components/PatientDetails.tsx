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
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });
};

export default function PatientDetails({ caller }: { caller: Caller }) {
  const [calls, setCalls] = useState<Call[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCall, setSelectedCall] = useState<Call | null>(null);

  useEffect(() => {
    const fetchCallerData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Fetch all calls for this caller
        const { data: callsData, error: callsError } = await supabase
          .from('calls')
          .select('*')
          .eq('phone_number', caller.phone_number)
          .order('call_timestamp', { ascending: false });

        if (callsError) throw new Error(callsError.message);

        // Deduplicate calls based on call_timestamp
        const uniqueCalls = callsData?.reduce((acc: Call[], current) => {
          const exists = acc.find(call => call.call_timestamp === current.call_timestamp);
          if (!exists) {
            acc.push(current);
          }
          return acc;
        }, []) || [];

        setCalls(uniqueCalls);
        
        // Set the most recent call as selected by default
        if (uniqueCalls.length > 0) {
          setSelectedCall(uniqueCalls[0]);
        }

        setIsLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred while fetching data');
        setIsLoading(false);
      }
    };

    let channel: ReturnType<typeof supabase.channel> | null = null;

    if (caller.phone_number) {
      fetchCallerData();

      // Create a single subscription
      channel = supabase
        .channel(`calls_${caller.phone_number}`)
        .on('postgres_changes', {
          event: 'INSERT',
          schema: 'public',
          table: 'calls',
          filter: `phone_number=eq.${caller.phone_number}`
        }, (payload) => {
          console.log('New call inserted:', payload);
          const newCall = payload.new as Call;
          setCalls(prevCalls => {
            if (prevCalls.some(call => call.call_timestamp === newCall.call_timestamp)) {
              return prevCalls;
            }
            return [newCall, ...prevCalls];
          });
          setSelectedCall(current => current || newCall);
        })
        .subscribe();
    }

    return () => {
      if (channel) {
        channel.unsubscribe();
      }
    };
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

  const getUrgencyColor = (score: number) => {
    if (score >= 8) return 'bg-red-500';
    if (score >= 5) return 'bg-orange-500';
    if (score >= 3) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  // Loading state UI
  if (isLoading) {
    return (
      <div className="grid gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-6">
              <div className="w-32 h-32 rounded-full bg-primary/10 animate-pulse" />
              <div className="space-y-4 flex-1">
                <div className="h-8 bg-primary/10 rounded animate-pulse w-1/3" />
                <div className="space-y-2">
                  <div className="h-4 bg-primary/10 rounded animate-pulse" />
                  <div className="h-4 bg-primary/10 rounded animate-pulse" />
                  <div className="h-4 bg-primary/10 rounded animate-pulse" />
                  <div className="h-4 bg-primary/10 rounded animate-pulse" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="h-[200px] bg-primary/10 rounded animate-pulse" />
            </CardContent>
          </Card>
          <Card className="col-span-2">
            <CardContent className="p-6">
              <div className="h-[200px] bg-primary/10 rounded animate-pulse" />
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardContent className="p-6">
            <div className="space-y-4">
              <div className="h-20 bg-primary/10 rounded animate-pulse" />
              <div className="h-20 bg-primary/10 rounded animate-pulse" />
              <div className="h-20 bg-primary/10 rounded animate-pulse" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Error state UI
  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center space-y-4">
            <div className="text-red-500 text-xl">Error Loading Data</div>
            <p className="text-muted-foreground">{error}</p>
            <Button 
              onClick={() => window.location.reload()} 
              variant="outline"
            >
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Main content UI (only shown when data is loaded)
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-6">
        <Card className="col-span-2">
          <CardContent className="p-6">
            <div className="flex items-center gap-6">
              <div className={`w-32 h-32 rounded-full flex items-center justify-center text-white text-4xl font-bold ${selectedCall ? getUrgencyColor(selectedCall.urgency_score) : 'bg-gray-300'}`}>
                {selectedCall ? selectedCall.urgency_score : '?'}
              </div>
              <div className="space-y-2 flex-1">
                <div className="flex justify-between items-start">
                  <h3 className="text-2xl font-bold">{caller.name || 'Anonymous'}</h3>
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
                <div className="space-y-1">
                  <p><span className="font-medium">Phone Number:</span> {caller.phone_number}</p>
                  <p><span className="font-medium">Last Contacted:</span> {formatDate(caller.last_call_timestamp)}</p>
                  <p><span className="font-medium">Previous History:</span> {caller.previous_history || 'None'}</p>
                  <p><span className="font-medium">Sexual Orientation:</span> {caller.sexual_orientation || 'Not specified'}</p>
                  {selectedCall && (
                    <p><span className="font-medium">Risk Level:</span> {
                      selectedCall.urgency_score >= 8 ? 'Critical - Immediate intervention needed' :
                      selectedCall.urgency_score >= 5 ? 'High Risk - Needs urgent attention' :
                      selectedCall.urgency_score >= 3 ? 'Moderate Risk - Monitor closely' :
                      'Low Risk - Stable but needs support'
                    }</p>
                  )}
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
              <span>Call from {formatDate(selectedCall.call_timestamp)}</span>
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