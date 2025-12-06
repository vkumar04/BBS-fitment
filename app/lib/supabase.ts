import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Database types
export interface Conversation {
  id: string;
  session_id: string;
  shop_domain: string;
  started_at: string;
  ended_at: string | null;
  message_count: number;
  sentiment_score: number;
  is_active: boolean;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  has_images: boolean;
  sentiment_score: number | null;
  timestamp: string;
}

export interface DailyMetric {
  id: string;
  date: string;
  shop_domain: string;
  total_conversations: number;
  active_conversations: number;
  total_messages: number;
  average_sentiment: number;
}
