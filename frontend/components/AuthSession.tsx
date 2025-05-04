import { useEffect, useState } from 'react';
import { supabase } from '../clients/supabaseClient';
import { Session } from '@supabase/supabase-js';

export default function useSupabaseSession() {

    const [session, setSession] = useState<Session | null>(null);
    const [user, setUser] = useState<Session["user"] | null>(null);
    const [loading, setLoading] = useState(true);


    useEffect(() => {
        const restoreSession = async () => {
            const { data } = await supabase.auth.getSession();
            console.log("Session from getSession():", data.session);
            setSession(data.session);
            setUser(data.session?.user ?? null);
            setLoading(false);
        };

        restoreSession();

        const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session);
            setUser(session?.user ?? null);
            console.log("Auth State Change Event:", _event, "New Session:", session); // <-- ADD THIS
        });

        return () => {
            authListener.subscription.unsubscribe();
        };
    }, []);

    return { session, user, loading };
}

