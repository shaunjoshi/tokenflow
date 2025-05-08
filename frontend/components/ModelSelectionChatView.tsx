import React, { useState, useRef, useEffect } from 'react';
import { AxiosInstance } from 'axios';
import { Session } from '@supabase/supabase-js';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Chip,
  Divider,
  Card,
  CardContent,
  IconButton,
  InputAdornment,
  Stack,
  Alert,
<<<<<<< Updated upstream
  useTheme
=======
  useTheme,
  PaperTypeMap,
>>>>>>> Stashed changes
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
<<<<<<< Updated upstream
=======
import ReactMarkdown from 'react-markdown';

// Define API Base URL (ensure it's available, e.g., from process.env)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000'; // Remove trailing slash if present
>>>>>>> Stashed changes

interface ModelSelectionChatViewProps {
  session: Session;
  apiClient: () => AxiosInstance | null;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  modelInfo?: {
    category: string;
    confidence: number;
    model: string;
    allCategories: Record<string, number>;
  };
}

const ModelSelectionChatView: React.FC<ModelSelectionChatViewProps> = ({ session, apiClient }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [categories] = useState(['reasoning', 'function-calling', 'text-to-text', 'multilingual', 'safety']);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    
    const client = apiClient();
    if (!client) {
      setError('API client not available. Please check your connection.');
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    try {
      // Send to model selection API
      const response = await client.post('/api/models/select', {
        prompt: inputText,
        possible_categories: categories,
        temperature: 0.7,
<<<<<<< Updated upstream
        top_p: 0.9
=======
        top_p: 0.9,
        max_tokens: 300 // Add max_tokens parameter
      };

      console.log("handleSubmit: Attempting fetch to /api/models/select with body:", requestBody);

      const response = await fetch(`${API_BASE_URL}/api/models/select`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestBody),
>>>>>>> Stashed changes
      });

      // Extract response data
      const { 
        completion, 
        prompt_category, 
        confidence_score, 
        selected_model,
        all_categories 
      } = response.data;

      // Add assistant response with model info
      const assistantMessage: Message = {
        id: Date.now().toString(),
        text: completion,
        sender: 'assistant',
        timestamp: new Date(),
        modelInfo: {
          category: prompt_category,
          confidence: confidence_score,
          model: selected_model,
          allCategories: all_categories
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error('API request failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to get response');
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to format model names for display
  const formatModelName = (modelId: string) => {
    const parts = modelId.split('/');
    if (parts.length >= 2) {
      const provider = parts[0];
      const model = parts[1].replace(/-/g, ' ');
      return `${provider} ${model}`;
    }
    return modelId;
  };

  // Get color for category chip
  const getCategoryColor = (category: string) => {
    const colorMap: Record<string, string> = {
<<<<<<< Updated upstream
      creative: theme.palette.success.main,
      factual: theme.palette.info.main,
      coding: theme.palette.warning.main,
      math: theme.palette.error.main,
      reasoning: theme.palette.secondary.main
=======
      reasoning: '#FF9800',    // Orange
      'function-calling': '#9C27B0', // Purple
      'text-to-text': '#4CAF50',    // Green
      multilingual: '#2196F3',  // Blue
      safety: '#F44336',     // Red
      // Keep old categories for backward compatibility
      creative: '#4CAF50', 
      factual: '#2196F3',  
      coding: '#9C27B0',   
      math: '#F44336'
>>>>>>> Stashed changes
    };
    return colorMap[category] || theme.palette.primary.main;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        Intelligent Model Selection Chat
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        Enter a prompt and see how it's classified and which AI model is selected for the response.
      </Typography>

      {/* Messages container */}
      <Paper 
        elevation={0}
        sx={{ 
          flexGrow: 1, 
          mb: 2, 
          p: 2, 
          overflowY: 'auto', 
          bgcolor: theme.palette.grey[50],
          height: '60vh',
          borderRadius: 2
        }}
      >
        {messages.length === 0 ? (
          <Box 
            sx={{ 
              height: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              flexDirection: 'column',
              color: 'text.secondary',
              p: 3
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 40, mb: 2, opacity: 0.7 }} />
            <Typography variant="body1">
              Send a message to see which model is best suited for your prompt
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Try different types of prompts like creative writing, factual questions, coding problems, math questions, or logical reasoning
            </Typography>
          </Box>
        ) : (
          messages.map((message) => (
            <Box 
              key={message.id} 
              sx={{ 
                display: 'flex', 
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                mb: 2
              }}
            >
              <Card 
                elevation={1}
                sx={{ 
                  maxWidth: '80%',
                  borderRadius: 2,
                  bgcolor: message.sender === 'user' 
                    ? theme.palette.primary.light
                    : theme.palette.background.paper
                }}
              >
<<<<<<< Updated upstream
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Typography 
                    variant="body1" 
                    sx={{ 
                      color: message.sender === 'user' 
                        ? theme.palette.primary.contrastText
                        : theme.palette.text.primary,
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {message.text}
                  </Typography>
                  
                  {message.modelInfo && (
                    <>
                      <Divider sx={{ my: 1.5 }} />
                      <Box sx={{ mt: 1 }}>
                        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                          <Chip 
                            label={`Category: ${message.modelInfo.category}`}
                            size="small"
                            sx={{ 
                              bgcolor: getCategoryColor(message.modelInfo.category),
                              color: '#fff',
                              fontWeight: 'bold'
                            }}
                          />
                          <Chip 
                            label={`Confidence: ${(message.modelInfo.confidence * 100).toFixed(0)}%`}
                            size="small"
                            variant="outlined"
                          />
                          <Chip 
                            label={`Model: ${formatModelName(message.modelInfo.model)}`}
                            size="small"
                            variant="outlined"
                            icon={<AutoAwesomeIcon fontSize="small" />}
                            sx={{ fontWeight: 'medium' }}
                          />
                        </Stack>

                        <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                          All Categories:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                          {Object.entries(message.modelInfo.allCategories)
                            .sort(([, a], [, b]) => b - a)
                            .map(([category, score]) => (
=======
                <Card 
                  elevation={1}
                  sx={{ 
                    maxWidth: '80%',
                    borderRadius: 2,
                    bgcolor: message.sender === 'user' 
                      ? theme.palette.primary.light
                      : theme.palette.background.paper
                  }}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    {message.sender === 'user' ? (
                      <Typography 
                        variant="body1" 
                        sx={{ 
                          color: theme.palette.primary.contrastText,
                          whiteSpace: 'pre-wrap' // Keep pre-wrap for user messages
                        }}
                      >
                        {message.text}
                      </Typography>
                    ) : (
                      // Use ReactMarkdown for assistant messages
                      <ReactMarkdown
                        components={{
                          p: ({node, ...props}) => <Typography variant="body2" paragraph sx={{mb:1}} {...props} />,
                          ul: ({node, ...props}) => <Box component="ul" sx={{ pl: 4, mt: 1, mb: 1 }} {...props} />,
                          ol: ({node, ...props}) => <Box component="ol" sx={{ pl: 4, mt: 1, mb: 1 }} {...props} />,
                          li: ({node, children, ...props}) => (
                            <li style={{ marginBottom: '4px' }}>
                              <Typography variant="body2" component="span" sx={{ display: 'inline', '& > p': { display: 'inline' } }} {...props}>
                                {children}
                              </Typography>
                            </li>
                          ),
                        }}
                      >{message.text}</ReactMarkdown>
                    )}
                    
                    {message.sender === 'assistant' && message.modelInfo && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Box sx={{ mt: 1 }}>
                          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                            {message.modelInfo?.prompt_category && (
>>>>>>> Stashed changes
                              <Chip
                                key={category}
                                label={`${category}: ${(score * 100).toFixed(0)}%`}
                                size="small"
                                variant="outlined"
                                sx={{ 
                                  fontSize: '0.7rem',
                                  height: 24,
                                  opacity: score > 0.1 ? 1 : 0.7
                                }}
                              />
                            ))
                          }
                        </Box>
                      </Box>
                    </>
                  )}
                </CardContent>
              </Card>
            </Box>
          ))
        )}
        <div ref={messagesEndRef} />
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Input area */}
      <Paper component="form" onSubmit={handleSubmit} elevation={2} sx={{ p: 2, borderRadius: 2 }}>
        <TextField
          fullWidth
          placeholder="Enter your prompt..."
          variant="outlined"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={isLoading}
          multiline
          minRows={1}
          maxRows={4}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton 
                  type="submit" 
                  color="primary" 
                  disabled={isLoading || !inputText.trim()}
                  edge="end"
                >
                  {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
                </IconButton>
              </InputAdornment>
            )
          }}
        />
      </Paper>
    </Box>
  );
};

export default ModelSelectionChatView; 