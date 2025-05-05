// src/app/page.tsx (Formerly layout.tsx)
"use client"; // REQUIRED: Keep this directive at the very top

import React from 'react';
// Ensure these paths are correct relative to src/app/page.tsx
import useSupabaseSession from '../components/AuthSession'; // Assuming it's in src/hooks/
import Auth from '../components/Auth';                      // Assuming it's in src/components/
import Dashboard from '../components/Dashboard';            // Assuming it's in src/components/
import { Container, Box, CircularProgress, Typography, Link } from '@mui/material';

// This component defines the content specifically for the "/" route
export default function Page() {
    const { session, loading } = useSupabaseSession();

    // --- Loading State ---
    // This part is good, handles waiting for the session check
    if (loading) {
        return (
            // Center loading indicator vertically and horizontally
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
                <CircularProgress />
                <Typography sx={{ ml: 2 }}>Loading Session...</Typography>
            </Box>
        );
    }

    // --- Main Content Rendering ---
    // The main content area for this specific page
    return (
        <Container component="main" maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            {/* Conditional rendering based on session state */}
            {!session ? (
                // Render the Authentication component if not logged in
                <Auth />
            ) : (
                // Render the Dashboard component if logged in
                // Passing session ensures Dashboard has the necessary user info
                <Dashboard key={session.user.id} session={session} />
            )}

            {/* Footer Placement Consideration:
             * This footer is currently specific to the HomePage.
             * If you want this footer on ALL pages, it's better to move
             * the <Box mt={5}...>...</Box> block into your app/layout.tsx
             * file, typically rendered after {children}.
             * Keeping it here is fine if it's ONLY for the home page. */}
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