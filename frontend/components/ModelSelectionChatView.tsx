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
  useTheme,
  PaperTypeMap,
  Link,
  alpha,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ReactMarkdown from 'react-markdown';

// Define API Base URL (ensure it's available, e.g., from process.env)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000'; // Remove trailing slash if present

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
    
    const token = session?.access_token;
    if (!token) {
      setError('Authentication token not available. Please log in again.');
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
    
    // Add placeholder assistant message
    const assistantId = Date.now().toString() + '-assistant';
    const placeholderMessage: Message = {
      id: assistantId,
      text: '',
      sender: 'assistant',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, placeholderMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    // Helper function to process individual SSE event blocks
    const processSseEventBlock = (block: string, targetAssistantId: string) => {
      console.log('[ModelSelectionChatView] Processing SSE block:', block);
      
      try {
        // Parse the JSON data directly since it's already in JSON format
        const eventData = JSON.parse(block.replace('data: ', ''));
        console.log('[ModelSelectionChatView] Parsed event data:', eventData);

        if (eventData.event === 'metadata' && eventData.data) {
          console.log('[ModelSelectionChatView] Processing metadata:', eventData.data);
          const newModelInfo = {
            category: eventData.data.prompt_category || 'text-to-text',
            confidence: eventData.data.confidence_score || 1.0,
            model: eventData.data.selected_model || eventData.data.model,
            allCategories: eventData.data.all_categories || {},
          };
          setMessages(prev =>
            prev.map(msg =>
              msg.id === targetAssistantId
                ? { ...msg, modelInfo: newModelInfo }
                : msg
            )
          );
        } else if (eventData.event === 'text_chunk' && eventData.data) {
          setMessages(prev => {
            console.log('[ModelSelectionChatView] Current messages:', prev);
            return prev.map(msg => {
              if (msg.id === targetAssistantId) {
                let currentText = msg.text || '';
                let newTextPortion = eventData.data;
                console.log('[ModelSelectionChatView] Current text:', currentText);
                console.log('[ModelSelectionChatView] New text portion:', newTextPortion);

                // If current text ends with a space and new portion also starts with a space,
                // append the new portion without its leading space to avoid double spaces.
                if (currentText.endsWith(' ') && newTextPortion.startsWith(' ')) {
                  currentText += newTextPortion.substring(1);
                } else {
                  currentText += newTextPortion;
                }
                console.log('[ModelSelectionChatView] Updated text:', currentText);
                return { ...msg, text: currentText };
              }
              return msg;
            });
          });
        } else if (eventData.event === 'error') {
          console.error('[ModelSelectionChatView] SSE Error Event:', eventData);
          throw new Error(eventData.data.detail || eventData.data.error || 'Unknown server error in stream');
        } else if (eventData.event === 'end_stream') {
          console.log("[ModelSelectionChatView] Received end_stream from backend.");
        }
      } catch (e) {
        console.error('[ModelSelectionChatView] Error processing event block:', e);
      }
    };

    try {
      const requestBody = {
        prompt: inputText,
        model: "llama-3.3-70b-versatile",
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 300
      };

      console.log('[ModelSelectionChatView] Sending request:', requestBody);
      console.log('[ModelSelectionChatView] API URL:', `${API_BASE_URL}/api/generate`);

      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestBody),
      });

      console.log('[ModelSelectionChatView] Response status:', response.status);
      console.log('[ModelSelectionChatView] Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[ModelSelectionChatView] Error response:', errorText);
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      if (!response.body) {
        console.error('[ModelSelectionChatView] Response body is null');
        throw new Error('Response body is null');
      }

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = '';

      console.log('[ModelSelectionChatView] Starting to read stream');

      while (true) {
        const { value, done } = await reader.read();
        console.log('[ModelSelectionChatView] Read chunk:', value);

        if (done) {
          console.log('[ModelSelectionChatView] Stream complete');
          if (buffer.trim()) {
            processSseEventBlock(buffer.trim(), assistantId);
          }
          break;
        }

        buffer += value;
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer

        for (const line of lines) {
          if (line.trim() && line.startsWith('data:')) {
            processSseEventBlock(line, assistantId);
          }
        }
      }
    } catch (err: any) {
      console.error('API request failed:', err);
      setError(err.message || 'Failed to get response');
      
      // Remove the placeholder message if there was an error
      setMessages(prev => prev.filter(msg => msg.id !== assistantId));
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
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  {message.sender === 'user' ? (
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        color: theme.palette.primary.contrastText,
                        whiteSpace: 'pre-wrap' 
                      }}
                    >
                      {message.text}
                    </Typography>
                  ) : (
                    <Box sx={{ color: 'text.primary', whiteSpace: 'pre-wrap' }}>
                      {message.text ? (
                        <ReactMarkdown
                          components={{
                            p: ({node, ...props}) => <Typography variant="body2" paragraph sx={{mb:1, whiteSpace: 'pre-wrap'}} {...props} />,
                            h1: ({node, ...props}) => <Typography variant="h1" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            h2: ({node, ...props}) => <Typography variant="h2" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            h3: ({node, ...props}) => <Typography variant="h3" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            h4: ({node, ...props}) => <Typography variant="h4" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            h5: ({node, ...props}) => <Typography variant="h5" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            h6: ({node, ...props}) => <Typography variant="h6" gutterBottom sx={{ whiteSpace: 'pre-wrap' }} {...props} />,
                            ul: ({node, ...props}) => <Box component="ul" sx={{ pl: 2, mt: 1, mb: 1 }} {...props} />,
                            ol: ({node, ...props}) => <Box component="ol" sx={{ pl: 2, mt: 1, mb: 1 }} {...props} />,
                            li: ({node, children, ...props}) => {
                              const { ref, ...restProps } = props as any;
                              return (
                                <Box component="li" sx={{ mb: 0.5, '& > p': { mb: 0.5 } }} {...restProps}>
                                  <Typography variant="body2" component="div">
                                    {children}
                                  </Typography>
                                </Box>
                              );
                            },
                            code: ({node, children, className, ...props}: any) => {
                              const match = /language-(\w+)/.exec(className || '');
                              const isInline = !match && !className;
                              return isInline 
                                ? <Typography component="code" variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: 1, fontSize: '0.875rem' }} {...props}>{children}</Typography>
                                : <Box component="pre" sx={{ fontFamily: 'monospace', bgcolor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100], p: 1.5, borderRadius: 1, overflowX: 'auto', fontSize: '0.875rem', my: 1 }} {...props}>{children}</Box>;
                            },
                            blockquote: ({node, ...props}) => (
                              <Box
                                component="blockquote"
                                sx={{
                                  pl: 2,
                                  ml: 0,
                                  mr: 0,
                                  my: 1.5,
                                  borderLeft: `4px solid ${theme.palette.grey[300]}`,
                                  bgcolor: alpha(theme.palette.primary.light, 0.05),
                                  '& > p': { mb: 0 },
                                  whiteSpace: 'pre-wrap'
                                }}
                                {...props}
                              />
                            ),
                            hr: ({node, ...props}) => <Divider sx={{ my: 2 }} {...props} />,
                            a: ({node, ...props}) => <Link {...props} target="_blank" rel="noopener noreferrer" />,
                          }}
                        >
                          {message.text}
                        </ReactMarkdown>
                      ) : (
                        <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                          Waiting for response...
                        </Typography>
                      )}
                    </Box>
                  )}
                  
                  {message.sender === 'assistant' && message.modelInfo && (
                    <>
                      <Divider sx={{ my: 1.5 }} />
                      <Box sx={{ mt: 1 }}>
                        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                          {message.modelInfo.category && (
                            <Chip
                              label={`Category: ${message.modelInfo.category}`}
                              size="small"
                              sx={{
                                bgcolor: getCategoryColor(message.modelInfo.category),
                                color: '#fff',
                                fontWeight: 'bold'
                              }}
                            />
                          )}
                          
                          {message.modelInfo.model && (
                            <Chip 
                              label={`Model: ${formatModelName(message.modelInfo.model)}`}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          )}
                          
                          {message.modelInfo.confidence && (
                            <Chip 
                              label={`Confidence: ${(message.modelInfo.confidence * 100).toFixed(0)}%`}
                              size="small"
                              color="secondary"
                              variant="outlined"
                            />
                          )}
                        </Stack>

                        {message.modelInfo.allCategories && Object.keys(message.modelInfo.allCategories).length > 0 && (
                          <>
                            <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                              All Categories:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {Object.entries(message.modelInfo.allCategories)
                                .sort(([, a], [, b]) => b - a)
                                .map(([category, score]) => (
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
                                ))}
                            </Box>
                          </>
                        )}
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