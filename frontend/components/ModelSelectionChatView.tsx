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
  useTheme
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

// Define API Base URL (ensure it's available, e.g., from process.env)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'; // Provide a fallback

interface ModelSelectionChatViewProps {
  session: Session;
  apiClient: () => AxiosInstance | null;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: number;
  modelInfo: ModelInfo | null;
}

interface ModelInfo {
  prompt_category: string;
  confidence_score: number;
  selected_model: string;
  all_categories: Record<string, number>;
}

const ModelSelectionChatView: React.FC<ModelSelectionChatViewProps> = ({ session, apiClient }) => {
  // Add log right at the start of the component function
  console.log("--- ModelSelectionChatView Component Render Start ---");

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [categories] = useState(['creative', 'factual', 'coding', 'math', 'reasoning']);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentAssistantText = useRef<string>('');
  const assistantMessageId = useRef<string | null>(null);
  const theme = useTheme();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Log messages state whenever it changes
  useEffect(() => {
    console.log("--- Messages state updated ---", messages);
  }, [messages]);

  const processEventBlock = (block: string) => {
    let eventType = '';
    let dataLines: string[] = []; // Store multiple data lines
    const lines = block.split('\n');

    for (const line of lines) {
        if (line.startsWith('event:')) {
            eventType = line.substring(6).trim();
        } else if (line.startsWith('data:')) {
            // Add the content of the data line (without 'data:')
            // Preserve leading/trailing spaces on individual lines for now
            dataLines.push(line.substring(5)); 
        }
    }
    
    // Join the data lines with newline characters as per SSE spec
    // Use trimEnd() to remove a potential trailing newline from the join, but preserve internal whitespace and leading spaces.
    const data = dataLines.join('\n').trimEnd(); 


    if (!eventType || !assistantMessageId.current) { // Check assistantId earlier
      // Allow processing 'data' even if empty (e.g. maybe an empty chunk means something?)
      // But skip if no eventType or no target message ID (unless it's a global error later)
      if(eventType !== 'error'){ // Allow 'error' event even without assistantId
         console.warn("Skipping event processing (no eventType or assistantId):", {eventType, hasData: !!data, assistantMsgId: assistantMessageId.current, block});
         return;
      }
    }
    // Check for empty data *after* joining, but allow metadata/end/error with potentially empty data payloads if needed
    if (!data && (eventType === 'text_chunk')) {
        console.warn("Skipping empty text_chunk:", {eventType, block});
        return;
    }

    console.log("Processing event:", eventType, "for message:", assistantMessageId.current ?? "N/A");


    // !!! Log the ID *before* conditional logic !!!
    const currentId = assistantMessageId.current; // Can be null for 'error'
    console.log(`[${eventType}] assistantMessageId = ${currentId ?? "null"}`);


    if (currentId && eventType === 'metadata') { // Check currentId exists for metadata
      try {
        const metadata = JSON.parse(data);
        console.log("METADATA EVENT PARSED:", metadata);
        const modelFromMeta = metadata?.selected_model;
        console.log("METADATA model value:", modelFromMeta, "(type:", typeof modelFromMeta, ")");

        setMessages(prev => {
          console.log(`Updating message ${currentId} with metadata.`);
          return prev.map(msg => {
            if (msg.id === currentId) {
              // Explicitly preserve the text from the previous state found in the map
              // Only add the modelInfo
              console.log(`[metadata] Preserving text for ${currentId}: "${msg.text}"`); 
              return { ...msg, text: msg.text, modelInfo: metadata }; 
            } else {
              return msg;
            }
          });
        });
      } catch (e) {
        console.error("Failed to parse metadata JSON:", e, data);
      }
    } else if (currentId && eventType === 'text_chunk') { // Check currentId exists for text_chunk
      
      setMessages(prev => {
         // Add detailed logging inside the state updater
         console.log(`[text_chunk] setMessages called. Prev state length: ${prev.length}`);
         const targetMsgExists = prev.some(m => m.id === currentId);
         console.log(`[text_chunk] Target message ${currentId} exists in prev state: ${targetMsgExists}`);
         
         const nextMessages = prev.map(msg => {
            if (msg.id === currentId) { // Use the captured currentId
              // REMOVED placeholder check, just append to current text or empty string
              const oldText = msg.text || ''; 
              const newText = oldText + data; 
              console.log(`[text_chunk] Updating msg ${currentId}. Old text was: "${msg.text}", New text: "${newText.substring(0,100)}..."`);
              return { ...msg, text: newText };
            } else {
              return msg;
            }
         });
         console.log(`[text_chunk] setMessages returning next state. Length: ${nextMessages.length}`);
         return nextMessages;
      });
    } else if (eventType === 'end_stream') {
      console.info(`[${eventType}] Processed end_stream.`);
      if (currentId) { // Only clear if we had an ID
          assistantMessageId.current = null;
          console.log(`[${eventType}] Cleared assistantMessageId.`);
      } else {
          console.log(`[${eventType}] No assistantMessageId to clear.`);
      }
      setIsLoading(false);
    } else if (eventType === 'error') {
      try {
        const errorData = JSON.parse(data);
        console.error('Processed error event:', errorData);
        setError(errorData.detail || errorData.error || 'Unknown backend error');
      } catch (e) {
        console.error("Failed to parse error JSON:", e, data);
        setError('Received unparsable error event from backend.');
      }
      assistantMessageId.current = null;
      console.log(`[${eventType}] Set assistantMessageId.current to null.`);
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    const userPrompt = inputText;
    setInputText('');
    setError(null);
    setIsLoading(true);
    currentAssistantText.current = '';

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      text: userPrompt,
      sender: 'user',
      timestamp: Date.now(),
      modelInfo: null
    };
    setMessages(prev => [...prev, userMessage]);

    // Add assistant placeholder message and store its ID
    const newAssistantId = `assistant-${Date.now()}`;
    assistantMessageId.current = newAssistantId;
    const placeholderMessage: Message = {
      id: newAssistantId,
      text: '', 
      sender: 'assistant',
      timestamp: Date.now(),
      modelInfo: null // Initialize as null
    };
    setMessages(prev => [...prev, placeholderMessage]);

    try {
      console.log("handleSubmit: Entering try block.");

      const token = session?.access_token;
      if (!token) throw new Error("Authentication token not found.");
      if (!API_BASE_URL) throw new Error("API Base URL not configured.");

      console.log(`handleSubmit: Token and URL found. API_BASE_URL: ${API_BASE_URL}`);

      const requestBody = {
        prompt: userPrompt,
        possible_categories: categories,
        temperature: 0.7,
        top_p: 0.9
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
      });

      console.log("handleSubmit: Fetch response received, status:", response.status, "ok:", response.ok);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("handleSubmit: Fetch response not OK. Status:", response.status, "Text:", errorText);
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
      }
      if (!response.body) {
         console.error("handleSubmit: Fetch response OK but response body is null.");
         throw new Error("Response body is null");
      }

      console.log("handleSubmit: Response body exists:", response.body);

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      console.log("handleSubmit: TextDecoderStream reader created.");

      let buffer = '';
      const boundaryRegex = /(\r?\n){2}/;

      console.log("handleSubmit: Starting stream processing outer loop...");
      while (true) {
        console.log("[Outer Loop Top] Buffer size:", buffer.length);
        let value: string | undefined;
        let done: boolean | undefined;
        try {
          console.log("[Outer Loop] Calling reader.read()...");
          const result = await reader.read();
          value = result.value;
          done = result.done;
          console.log("[Outer Loop] reader.read() returned:", { value: value ? value.substring(0,50)+"..." : null, done });
        } catch (readError) {
          console.error("[Outer Loop] Error during reader.read():", readError);
          setError(`Error reading stream: ${readError}`);
          setIsLoading(false);
          assistantMessageId.current = null;
          break; // Exit loop on read error
        }

        if (value) {
            buffer += value;
        }

        console.log("[Processing Buffer] Size before processing:", buffer.length);
        let processedSomethingInLoop = false; // Flag to see if inner loop ran

        // Process all complete blocks in the current buffer
        while (true) { // Inner loop to process buffer contents
            console.log("[Inner Loop Start] Current Buffer:", buffer.substring(0, 100) + (buffer.length > 100 ? "..." : "")); // Log buffer start
            const boundaryMatch = buffer.match(boundaryRegex);
            
            if (!boundaryMatch || boundaryMatch.index === undefined) {
                console.log("[Inner Loop End] No complete event block boundary found in buffer.");
                break; // Exit inner loop, wait for more data or done
            }
            
            processedSomethingInLoop = true; // Mark that we are processing
            const boundaryIndex = boundaryMatch.index;
            const boundaryLength = boundaryMatch[0].length;
            const eventBlock = buffer.substring(0, boundaryIndex);
            const remainingBuffer = buffer.substring(boundaryIndex + boundaryLength); // Store remaining part
            buffer = remainingBuffer; // Update buffer *after* extracting block

            console.log("[Process Block] Extracted:", eventBlock.replace(/\n/g, "\\n")); // Log extracted block clearly
            console.log("[Process Block] Remaining Buffer:", buffer.substring(0, 100) + (buffer.length > 100 ? "..." : "")); // Log remaining buffer
            processEventBlock(eventBlock);

            // Yield to the event loop to allow React to render updates
            // before processing the next block in the buffer.
            await new Promise(resolve => setTimeout(resolve, 0));

            // Check if isLoading changed state (e.g., due to end_stream or error)
            // Need to read isLoading state directly, as it might have changed within processEventBlock
            // This check might be tricky due to state update timing. Logging inside processEventBlock is more reliable.
            // if (!isLoading) { 
            //     console.log("[Inner Loop Exit] isLoading became false (end/error event processed).");
            //     break; 
            // }
        }
        console.log("[Processing Buffer] Size after inner loop:", buffer.length, "Processed Block?:", processedSomethingInLoop);

        if (!isLoading) {
            console.log("[Outer Loop Exit] isLoading is false.");
            break; // Exit outer loop if end/error occurred
        }

        if (done) {
          console.info('[Outer Loop End] Stream done=true.');
          if (buffer.trim().length > 0) {
             console.log("[Final Process] Processing remaining buffer:", buffer);
             processEventBlock(buffer.trim());
          }
          if (assistantMessageId.current) {
              console.warn("Stream finished reading, but end_stream/error event might not have been received/processed.");
              setIsLoading(false);
              assistantMessageId.current = null;
          }
          break;
        }
      } // End outer while(true)
      
      console.log("[Stream End] Releasing reader lock if not closed.");
      if (!reader.closed) {
          reader.releaseLock();
      }

    } catch (err: any) {
      console.error('handleSubmit: Error caught in catch block:', err);
      setError(err.message || 'Failed to get streaming response');
      if (assistantMessageId.current) {
          setMessages(prev => prev.filter(msg => msg.id !== assistantMessageId.current));
      }
      setIsLoading(false);
      assistantMessageId.current = null;
    }
  };

  // Helper to format model names for display
  const formatModelName = (modelId: string | undefined | null): string => {
    if (!modelId || typeof modelId !== 'string') {
      return 'Unknown Model';
    }
    
    try {
      // Handle common model ID formats like "organization/model-name:version"
      const parts = modelId.split('/');
      if (parts.length > 1) {
        // Extract just the model name part
        let modelName = parts[1];
        
        // Remove version suffix if present (after colon)
        if (modelName.includes(':')) {
          modelName = modelName.split(':')[0];
        }
        
        // Format the model name (replace hyphens with spaces, capitalize)
        return modelName
          .replace(/-/g, ' ')
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
      }
      
      // Fallback for simple model names without organization prefix
      return modelId
        .replace(/-/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    } catch (error) {
      console.error('Error formatting model name:', error);
      return modelId; // Return original if formatting fails
    }
  };

  // Get color for category chip
  const getCategoryColor = (category: string): string => {
    const colorMap: Record<string, string> = {
      creative: '#4CAF50', // Green
      factual: '#2196F3',  // Blue
      coding: '#9C27B0',   // Purple
      math: '#F44336',     // Red
      reasoning: '#FF9800' // Orange
    };
    return colorMap[category.toLowerCase()] || '#757575'; // Grey default
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
          messages.map((message) => {
            // Log the specific modelInfo being rendered
            console.log('Rendering message:', message.id, 'modelInfo:', message.modelInfo);
            return (
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
                    
                    {message.sender === 'assistant' && message.modelInfo && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Box sx={{ mt: 1 }}>
                          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                            {message.modelInfo?.prompt_category && (
                              <Chip 
                                label={`Category: ${message.modelInfo.prompt_category}`}
                                size="small"
                                sx={{ 
                                  bgcolor: getCategoryColor(message.modelInfo.prompt_category),
                                  color: '#fff',
                                  fontWeight: 'bold'
                                }}
                              />
                            )}
                            
                            {message.modelInfo?.selected_model && (
                              <Chip 
                                label={`Model: ${formatModelName(message.modelInfo.selected_model)}`}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            )}
                            
                            {message.modelInfo?.confidence_score && (
                              <Chip 
                                label={`Confidence: ${(message.modelInfo.confidence_score * 100).toFixed(0)}%`}
                                size="small"
                                color="secondary"
                                variant="outlined"
                              />
                            )}
                          </Stack>

                          {message.modelInfo?.all_categories && Object.keys(message.modelInfo.all_categories).length > 0 && (
                            <>
                              <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                                All Categories:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                {Object.entries(message.modelInfo.all_categories)
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
                                  ))
                                }
                              </Box>
                            </>
                          )}
                        </Box>
                      </>
                    )}
                  </CardContent>
                </Card>
              </Box>
            );
          })
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