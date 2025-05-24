import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://hlizjdprcsnfjwtxerhl.supabase.co';
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
});

export { supabase };
