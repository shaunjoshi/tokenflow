// src/components/Auth.tsx
import { useState } from 'react';
import {
  Box,
  Button,
  Paper,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  Fade,
  Divider,
  Stack,
} from '@mui/material';
import { supabase } from '../clients/supabaseClient';
import Image from 'next/image';
import GoogleIcon from '@mui/icons-material/Google';
import GitHubIcon from '@mui/icons-material/GitHub';

export default function Auth() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAuth = async () => {
    setLoading(true);
    setMessage(null);

    let error;
    if (isSignUp) {
      ({ error } = await supabase.auth.signUp({ email, password }));
    } else {
      let data;
      ({ data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      }));
    }

    if (error) {
      setMessage({ type: 'error', text: error.message });
    } else {
      setMessage({
        type: 'success',
        text: isSignUp
          ? 'Sign up successful! Check your email if confirmation is enabled.'
          : 'Logged in successfully!',
      });
    }

    setLoading(false);
  };

  const handleOAuth = async (provider: 'google' | 'github') => {
    const { error } = await supabase.auth.signInWithOAuth({ provider });
    if (error) {
      setMessage({ type: 'error', text: error.message });
    }
  };

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      bgcolor="#f5f5f5"
    >
      <Fade in timeout={500}>
        <Paper elevation={4} sx={{ p: 4, width: '100%', maxWidth: 420 }}>
          <Box textAlign="center" mb={3}>
            {/* Logo or Illustration */}
            <Image
              src="/logo.svg" // Replace with your own image path
              alt="Alkme.ai Logo"
              width={60}
              height={60}
            />
            <Typography variant="h5" fontWeight="bold" mt={2}>
              {isSignUp ? 'Create a new account' : 'Welcome back'}
            </Typography>
            <Typography variant="body2" mt={1}>
              {isSignUp
                ? 'Sign up to start chatting with your documents.'
                : 'Log in to your alkme.ai account.'}
            </Typography>
          </Box>

          <Stack
            component="form"
            spacing={2}
            onSubmit={e => e.preventDefault()}
          >
            <TextField
              label="Email"
              type="email"
              fullWidth
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
            />
            <TextField
              label="Password"
              type="password"
              fullWidth
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />

            <Button
              variant="contained"
              color="primary"
              onClick={handleAuth}
              disabled={loading}
              fullWidth
              size="large"
            >
              {loading ? (
                <CircularProgress size={24} sx={{ color: 'white' }} />
              ) : isSignUp ? (
                'Sign Up'
              ) : (
                'Log In'
              )}
            </Button>

            <Button
              onClick={() => setIsSignUp(!isSignUp)}
              color="secondary"
              fullWidth
            >
              {isSignUp
                ? 'Already have an account? Log In'
                : 'New here? Create an account'}
            </Button>
          </Stack>

          <Divider sx={{ my: 3 }}>or continue with</Divider>

          {/* OAuth Buttons */}
          <Stack spacing={1}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<GoogleIcon />}
              onClick={() => handleOAuth('google')}
            >
              Continue with Google
            </Button>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<GitHubIcon />}
              onClick={() => handleOAuth('github')}
            >
              Continue with GitHub
            </Button>
          </Stack>

          {message && (
            <Alert severity={message.type} sx={{ mt: 3 }}>
              {message.text}
            </Alert>
          )}
        </Paper>
      </Fade>
    </Box>
  );
}
