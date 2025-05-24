import React, { useState, useCallback, useRef, useEffect } from 'react';
import { AxiosInstance } from 'axios';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Grid,
  Slider,
  Alert,
  Stack,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Divider,
  Card,
  CardContent,
  useTheme,
} from '@mui/material';
import ReactMarkdown from 'react-markdown'; // For rendering LLM output
import SendIcon from '@mui/icons-material/Send';
import DifferenceIcon from '@mui/icons-material/Difference'; // Uncommented Icon

// --- Diff Imports --- Uncommented
import { Diff, Hunk } from 'react-diff-view'; 
import { diffLines, createPatch } from 'diff'; 
import 'react-diff-view/style/index.css'; 

// --- Available Models (Groq-supported models) ---
const AVAILABLE_MODELS = [
  { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B Versatile' },
  { id: 'llama-3.1-8b-instant', name: 'Llama 3.1 8B Instant' },
  { id: 'gemma2-9b-it', name: 'Gemma 2 9B' },
];

// Define the structure for the API response (mirroring backend's CompressionResponse)
interface CompressionResult {
  original_text: string;
  compressed_text: string;
  original_tokens: number;
  compressed_tokens: number;
  compression_ratio: number;
}

interface PromptCompressionViewProps {
  apiClient: () => AxiosInstance | null;
}

// Helper type for the diff view
interface ParsedDiffFile {
  hunks: any[];
  oldPath: string;
  newPath: string;
}

const PromptCompressionView: React.FC<PromptCompressionViewProps> = ({ apiClient }) => {
  // --- State for Compression ---
  const [inputText, setInputText] = useState<string>('');
  const [compressionRatio, setCompressionRatio] = useState<number>(0.5);
  const [isCompressing, setIsCompressing] = useState<boolean>(false);
  const [compressionError, setCompressionError] = useState<string | null>(null);
  const [compressionResult, setCompressionResult] = useState<CompressionResult | null>(null);

  // --- State for Generation ---
  const [selectedModelOriginal, setSelectedModelOriginal] = useState<string>(AVAILABLE_MODELS[0]?.id || '');
  const [selectedModelCompressed, setSelectedModelCompressed] = useState<string>(AVAILABLE_MODELS[0]?.id || '');
  const [isGeneratingOriginal, setIsGeneratingOriginal] = useState<boolean>(false);
  const [isGeneratingCompressed, setIsGeneratingCompressed] = useState<boolean>(false);
  const [generationErrorOriginal, setGenerationErrorOriginal] = useState<string | null>(null);
  const [generationErrorCompressed, setGenerationErrorCompressed] = useState<string | null>(null);
  const [outputOriginal, setOutputOriginal] = useState<string>('');
  const [outputCompressed, setOutputCompressed] = useState<string>('');
  
  // --- State for Diff View ---
  const [diffData, setDiffData] = useState<ParsedDiffFile | null>(null);
  const [showDiff, setShowDiff] = useState<boolean>(false);

  // Refs to manage streaming state for each generation box independently
  const currentOutputStreamRef = useRef<((text: string) => void) | null>(null);
  const currentStreamLoadingRef = useRef<((loading: boolean) => void) | null>(null);
  const currentStreamErrorRef = useRef<((error: string | null) => void) | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<string> | null>(null);
  const theme = useTheme(); // Get theme for styling

  // Generate diff when both outputs are available
  useEffect(() => {
    if (outputOriginal && outputCompressed) {
      generateDiff();
    }
  }, [outputOriginal, outputCompressed]);

  // Function to generate diff between original and compressed outputs
  const generateDiff = useCallback(() => {
    try {
      // Generate diff between the two outputs
      const changes = diffLines(outputOriginal, outputCompressed);
      
      // Create a patch from the diff
      const patch = createPatch(
        'original.txt',
        outputOriginal,
        outputCompressed,
        'Original Output',
        'Compressed Output'
      );
      
      // Parse the patch to get hunks
      const hunks = changes.map((change, i) => ({
        content: change.value,
        oldStart: i + 1,
        newStart: i + 1,
        oldLines: change.removed ? change.count : 0,
        newLines: change.added ? change.count : 0,
        changes: [
          {
            content: change.value,
            type: change.added ? 'insert' : (change.removed ? 'delete' : 'normal'),
            oldLineNumber: change.removed ? i + 1 : null,
            newLineNumber: change.added ? i + 1 : null,
          }
        ],
      }));
      
      // Set the diff data
      setDiffData({
        hunks,
        oldPath: 'Original Output',
        newPath: 'Compressed Output',
      });
      
      setShowDiff(true);
    } catch (error) {
      console.error('Error generating diff:', error);
    }
  }, [outputOriginal, outputCompressed]);

  // --- Compression Logic ---
  const handleCompress = useCallback(async () => {
    const client = apiClient();
    if (!client || !inputText.trim()) {
      setCompressionError('API client not available or input text is empty.');
      return;
    }

    setIsCompressing(true);
    setCompressionError(null);
    setCompressionResult(null);
    setOutputOriginal(''); // Clear previous generation results
    setOutputCompressed('');
    setGenerationErrorOriginal(null);
    setGenerationErrorCompressed(null);
    setDiffData(null); // Clear previous diff data
    setShowDiff(false);

    try {
      console.log(`Sending text for compression. Target ratio: ${compressionRatio}`);
      const response = await client.post<CompressionResult>('/api/compress', {
        text: inputText,
        ratio: compressionRatio,
      });
      console.log("Compression response received:", response.data);
      setCompressionResult(response.data);
    } catch (err: any) {
      console.error("Compression error:", err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to compress text.';
      setCompressionError(errorMsg);
    } finally {
      setIsCompressing(false);
    }
  }, [apiClient, inputText, compressionRatio]);

  // --- Generation Logic (Handles SSE Stream) ---
  const handleGenerate = useCallback(async (
    promptText: string | undefined,
    modelId: string,
    setOutput: (text: string) => void,
    setIsLoading: (loading: boolean) => void,
    setError: (error: string | null) => void
  ) => {
    if (!promptText || !modelId) {
      setError('Prompt text or model ID is missing.');
      return;
    }
    
    // Ensure any previous stream is closed
    readerRef.current?.cancel('Starting new generation').catch(() => {}); // Cancel previous read if active
    readerRef.current = null;

    setIsLoading(true);
    setError(null);
    setOutput(''); // Clear previous output
    let accumulatedText = '';

    try {
      const token = (apiClient()?.defaults?.headers as any)?.Authorization?.split(' ')[1];
      if (!token) {
        throw new Error('No authentication token available');
      }

      console.log(`[PromptCompressionView] Sending request for model ${modelId}`);
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '');
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          prompt: promptText,
          model: modelId,
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 1000
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      readerRef.current = reader;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const lines = value.split('\n');
        for (const line of lines) {
          if (line.trim() && line.startsWith('data:')) {
            try {
              const eventData = JSON.parse(line.replace('data: ', ''));
              console.log('[PromptCompressionView] Received event:', eventData);

              if (eventData.event === 'text_chunk' && eventData.data) {
                accumulatedText += eventData.data;
                setOutput(accumulatedText);
              } else if (eventData.event === 'error') {
                throw new Error(eventData.data.detail || eventData.data.error || 'Unknown error');
              }
            } catch (e) {
              console.error('[PromptCompressionView] Error processing event:', e);
              setError(e instanceof Error ? e.message : 'Error processing stream');
            }
          }
        }
      }
    } catch (err: any) {
      console.error('[PromptCompressionView] Generation error:', err);
      setError(err.message || 'Failed to generate text');
    } finally {
      setIsLoading(false);
      readerRef.current = null;
    }
  }, [apiClient]);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h5" gutterBottom>
        Prompt Compression & Generation Demo
      </Typography>
      <Typography variant="body1" paragraph color="text.secondary">
        Compress text using LLMLingua, then generate responses from the original and compressed versions using a selected LLM, and see the difference.
      </Typography>

      {/* --- Compression Section --- */}
      <Paper elevation={2} sx={{ p: 3, mb: 4, border: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>1. Compress Prompt</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              label="Text to Compress"
              multiline
              rows={8} // Increased rows
              fullWidth
              variant="outlined"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isCompressing}
            />
          </Grid>
          <Grid item xs={12} md={6}>
             <Typography gutterBottom id="compression-ratio-slider-label">Compression Ratio (target)</Typography>
             <Slider
               value={compressionRatio}
               onChange={(event, newValue) => setCompressionRatio(newValue as number)}
               aria-labelledby="compression-ratio-slider-label"
               valueLabelDisplay="auto"
               step={0.05}
               marks
               min={0.1}
               max={1.0}
               disabled={isCompressing}
             />
          </Grid>
          <Grid item xs={12} md={6} sx={{ display: 'flex', alignItems: 'center', pt: { md: 3.5} }}> {/* Align button better */}
             <Button
               variant="contained"
               color="primary"
               onClick={handleCompress}
               disabled={isCompressing || !inputText.trim()}
               startIcon={isCompressing ? <CircularProgress size={20} color="inherit" /> : null}
               fullWidth
             >
               {isCompressing ? 'Compressing...' : 'Compress Text'}
             </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Compression Error Display */}
      {compressionError && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setCompressionError(null)}>
          {compressionError}
        </Alert>
      )}

      {/* --- Results & Generation Section --- */}
      {compressionResult && (
        <Box sx={{ mt: 4 }}>
           <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>2. Generate from Prompts</Typography>
           <Grid container spacing={4} >
            {/* --- Original Prompt Generation Box --- */}
            <Grid item xs={12} md={6}>
               <Card variant="outlined" sx={{ height: '100%' }}> {/* Wrap in Card */}
                 <CardContent>
                    <Typography variant="subtitle1" gutterBottom>Original Prompt ({compressionResult.original_tokens} tokens)</Typography>
                    <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '200px', overflowY: 'auto', whiteSpace: 'pre-wrap', backgroundColor: theme.palette.grey[100] }}>
                      {compressionResult.original_text}
                    </Paper>
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel id="original-model-select-label">Select LLM</InputLabel>
                      <Select
                        labelId="original-model-select-label"
                        value={selectedModelOriginal}
                        label="Select LLM"
                        onChange={(e: SelectChangeEvent) => setSelectedModelOriginal(e.target.value as string)}
                        disabled={isGeneratingOriginal}
                      >
                        {AVAILABLE_MODELS.map(model => (
                          <MenuItem key={model.id} value={model.id}>{model.name}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Button
                      variant="outlined"
                      color="secondary"
                      fullWidth
                      onClick={() => handleGenerate(
                          compressionResult.original_text,
                          selectedModelOriginal,
                          setOutputOriginal,
                          setIsGeneratingOriginal,
                          setGenerationErrorOriginal
                      )}
                      disabled={isGeneratingOriginal || !selectedModelOriginal}
                      startIcon={isGeneratingOriginal ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                      sx={{ mb: 2 }}
                    >
                      {isGeneratingOriginal ? 'Generating...' : 'Generate with Original'}
                    </Button>
                    {generationErrorOriginal && (
                      <Alert severity="error" sx={{ mb: 2 }} onClose={() => setGenerationErrorOriginal(null)}>
                        {generationErrorOriginal}
                      </Alert>
                    )}
                    {outputOriginal && (
                       <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>Generated Output:</Typography>
                           <Paper variant="outlined" sx={{ p: 2, maxHeight: '400px', overflowY: 'auto', whiteSpace: 'pre-wrap', backgroundColor: theme.palette.grey[50] }}>
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
                             >{outputOriginal}</ReactMarkdown>
                           </Paper>
                       </Box>
                    )}
                 </CardContent>
               </Card>
             </Grid>

            {/* --- Compressed Prompt Generation Box --- */}
            <Grid item xs={12} md={6}>
              <Card variant="outlined" sx={{ height: '100%' }}> {/* Wrap in Card */}
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                     Compressed Prompt ({compressionResult.compressed_tokens} tokens, Achieved Ratio: {compressionResult.compression_ratio.toFixed(2)})
                   </Typography>
                   <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '200px', overflowY: 'auto', whiteSpace: 'pre-wrap', backgroundColor: theme.palette.primary.light }}> 
                     {compressionResult.compressed_text}
                   </Paper>
                   <FormControl fullWidth sx={{ mb: 2 }}>
                     <InputLabel id="compressed-model-select-label">Select LLM</InputLabel>
                     <Select
                        labelId="compressed-model-select-label"
                        value={selectedModelCompressed}
                        label="Select LLM"
                        onChange={(e: SelectChangeEvent) => setSelectedModelCompressed(e.target.value as string)}
                        disabled={isGeneratingCompressed}
                      >
                       {AVAILABLE_MODELS.map(model => (
                         <MenuItem key={model.id} value={model.id}>{model.name}</MenuItem>
                       ))}
                     </Select>
                   </FormControl>
                   <Button
                     variant="outlined"
                     color="secondary"
                     fullWidth
                     onClick={() => handleGenerate(
                         compressionResult.compressed_text,
                         selectedModelCompressed,
                         setOutputCompressed,
                         setIsGeneratingCompressed,
                         setGenerationErrorCompressed
                     )}
                     disabled={isGeneratingCompressed || !selectedModelCompressed || !compressionResult.compressed_text}
                     startIcon={isGeneratingCompressed ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
                     sx={{ mb: 2 }}
                   >
                     {isGeneratingCompressed ? 'Generating...' : 'Generate with Compressed'}
                   </Button>
                   {generationErrorCompressed && (
                     <Alert severity="error" sx={{ mb: 2 }} onClose={() => setGenerationErrorCompressed(null)}>
                       {generationErrorCompressed}
                     </Alert>
                   )}
                   {outputCompressed && (
                       <Box sx={{ mt: 2 }}>
                           <Typography variant="subtitle2" gutterBottom>Generated Output:</Typography>
                           <Paper variant="outlined" sx={{ p: 2, maxHeight: '400px', overflowY: 'auto', whiteSpace: 'pre-wrap', backgroundColor: theme.palette.grey[50] }}>
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
                              >{outputCompressed}</ReactMarkdown>
                           </Paper>
                       </Box>
                   )}
                 </CardContent>
               </Card>
             </Grid>
           </Grid>
           
           {/* --- Diff View Section --- */}
           {outputOriginal && outputCompressed && (
             <Box sx={{ mt: 4 }}>
               <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                 3. Compare Outputs (Diff View)
               </Typography>
               <Paper elevation={3} sx={{ p: 3, mb: 4, border: `1px solid ${theme.palette.divider}` }}>
                 <Button
                   variant="outlined"
                   color="primary"
                   onClick={() => setShowDiff(!showDiff)}
                   startIcon={<DifferenceIcon />}
                   sx={{ mb: 2 }}
                 >
                   {showDiff ? 'Hide Diff' : 'Show Diff'}
                 </Button>
                 
                 {showDiff && diffData && (
                   <Box sx={{ 
                     mt: 2, 
                     border: `1px solid ${theme.palette.divider}`,
                     fontSize: '0.875rem',
                     lineHeight: 1.4,
                     fontFamily: '"Roboto Mono", monospace',
                     '& .diff-gutter-col': { width: '50px' },
                   }}>
                     <Diff
                       viewType="split"
                       diffType="modify"
                       hunks={diffData.hunks}
                     >
                       {hunks => hunks.map(hunk => (
                         <Hunk key={hunk.content} hunk={hunk} />
                       ))}
                     </Diff>
                   </Box>
                 )}
               </Paper>
             </Box>
           )}
        </Box>
      )}
    </Box>
  );
};

export default PromptCompressionView; 