import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseKey);

export type Caller = {
  phone_number: string;
  name: string | null;
  aggregated_transcript: string | null;
  last_call_timestamp: string;
  previous_history?: string;
  sexual_orientation?: string;
  urgency?: number;
};

export type Call = {
  id: string;
  phone_number: string;
  call_timestamp: string;
  audio_url: string;
  transcript: string;
  duration: number;
  sentiment?: {
    score: number;
    labels: string[];
  };
}; 