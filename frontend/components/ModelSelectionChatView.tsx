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

  // Helper function to preprocess markdown text for better parsing
  const preprocessMarkdown = (text: string): string => {
    if (!text) return '';
    let processedText = text;

    // Normalize bold: ** text ** -> **text**
    processedText = processedText.replace(/\*\*\s*([\s\S]+?)\s*\*\*/g, '**$1**');

    // Normalize italic (underscore): _ text _ -> _text_
    processedText = processedText.replace(/_\s*([\s\S]+?)\s*_/g, '_$1_');
    
    // Normalize strikethrough: ~~ text ~~ -> ~~text~~
    processedText = processedText.replace(/~~\s*([\s\S]+?)\s*~~/g, '~~$1~~');

    // For list items, ensure only one space after the marker (e.g. "*  item" becomes "* item")
    processedText = processedText.split('\n').map(line => {
      // Matches "*", "-", or "1." followed by two or more spaces at the start of a line (potentially with leading whitespace)
      return line.replace(/^(\s*([*-]|\d+\.)\s)\s+/g, '$1');
    }).join('\n');

    return processedText;
  };

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
      let eventType = '';
      let eventData = '';
      const lines = block.split('\n');

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.substring(6).trim();
        } else if (line.startsWith('data:')) {
          eventData = line.substring(5);
        }
      }

      // console.log(`[ModelSelectionChatView] Processed Event: ${eventType}, Data: ${eventData.substring(0, 100)}`);

      if (eventType === 'metadata' && eventData) {
        try {
          const metadata = JSON.parse(eventData);
          const newModelInfo = {
            category: metadata.prompt_category,
            confidence: metadata.confidence_score,
            model: metadata.selected_model,
            allCategories: metadata.all_categories,
          };
          setMessages(prev =>
            prev.map(msg =>
              msg.id === targetAssistantId
                ? { ...msg, modelInfo: newModelInfo }
                : msg
            )
          );
        } catch (e) {
          console.error('[ModelSelectionChatView] Error parsing metadata:', e);
        }
      } else if (eventType === 'text_chunk' && eventData !== undefined) {
        setMessages(prev =>
          prev.map(msg => {
            if (msg.id === targetAssistantId) {
              let currentText = msg.text || '';
              let newTextPortion = eventData;

              // If current text ends with a space and new portion also starts with a space,
              // append the new portion without its leading space to avoid double spaces.
              if (currentText.endsWith(' ') && newTextPortion.startsWith(' ')) {
                currentText += newTextPortion.substring(1);
              } else {
                currentText += newTextPortion;
              }
              return { ...msg, text: currentText };
            }
            return msg;
          })
        );
      } else if (eventType === 'error' && eventData) {
        console.error('[ModelSelectionChatView] SSE Error Event:', eventData);
        try {
          const errorData = JSON.parse(eventData);
          throw new Error(errorData.detail || errorData.error || 'Unknown server error in stream');
        } catch (e: any) {
          throw new Error('Error in stream: ' + (e.message || eventData || 'Unknown error'));
        }
      } else if (eventType === 'end_stream') {
        console.log("[ModelSelectionChatView] Received end_stream from backend.");
      }
    };

    try {
      const requestBody = {
        prompt: inputText,
        possible_categories: categories,
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 300
      };

      const response = await fetch(`${API_BASE_URL}/api/models/select`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = '';
      const boundaryRegex = /(\r?\n){2}/; // Matches \n\n or \r\n\r\n

      while (true) {
        const { value, done } = await reader.read();

        if (done) {
          console.log('[ModelSelectionChatView] Stream complete');
          if (buffer.trim()) {
            processSseEventBlock(buffer.trim(), assistantId);
          }
          break;
        }

        buffer += value;
        let match;
        // Process buffer for complete event blocks
        while ((match = buffer.match(boundaryRegex)) && match.index !== undefined) {
          const eventBlock = buffer.substring(0, match.index);
          buffer = buffer.substring(match.index + match[0].length);
          if (eventBlock.trim()) {
            processSseEventBlock(eventBlock, assistantId);
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
                          {preprocessMarkdown(message.text)}
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