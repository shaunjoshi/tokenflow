// src/app/page.tsx (Formerly App.tsx)
"use client"; // <-- ADD THIS DIRECTIVE AT THE TOP

import React from 'react';
import useSupabaseSession from '../components/AuthSession'; // Adjust path if needed
import Auth from '../components/Auth'; // Adjust path if needed
import Dashboard from '../components/Dashboard'; // Adjust path if needed
import { Container, Box, CircularProgress, Typography, Link } from '@mui/material';

// The component definition remains largely the same, just named implicitly "Page" by Next.js
export default function Page() {
    const { session, loading } = useSupabaseSession();

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <CircularProgress />
                <Typography sx={{ ml: 2 }}>Loading Session...</Typography>
            </Box>
        );
    }

    // The Container should likely be inside the return, not wrapping everything
    // Let the RootLayout handle the main page structure potentially
    return (
        <Container component="main" maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            {!session ? (
                <Auth />
            ) : (
                <Dashboard key={session.user.id} session={session} />
            )}
            {/* Footer can stay or move to layout */}
            <Box mt={5} mb={2} textAlign="center">
                <Typography variant="body2" color="text.secondary">
                    MVP - Built for demonstration. | Need help?{' '}
                    <Link href="#" color="inherit">
                        Contact Support
                    </Link>
                </Typography>
            </Box>
        </Container>
    );
}