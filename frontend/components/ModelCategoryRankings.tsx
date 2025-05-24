import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Collapse,
  IconButton,
  Divider,
  Chip,
  List,
  ListItem,
  ListItemText,
  useTheme,
  CircularProgress,
  Alert,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

// API Base URL from environment
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') ||
  'http://localhost:8000';

// Type definitions for model rankings data
interface ModelInfo {
  id: string;
  name: string;
  primary: boolean;
}

interface CategoryData {
  title: string;
  description: string;
  models: ModelInfo[];
}

interface ModelRankingsData {
  [category: string]: CategoryData;
}

// Get color for category
const getCategoryColor = (category: string): string => {
  const colorMap: Record<string, string> = {
    reasoning: '#FF9800',
    'function-calling': '#9C27B0',
    'text-to-text': '#4CAF50',
    multilingual: '#2196F3',
    nsfw: '#F44336',
    safety: '#F44336', // Keeping for backward compatibility
  };
  return colorMap[category] || '#757575';
};

interface ModelCategoryRankingsProps {
  initialExpanded?: boolean;
}

const ModelCategoryRankings: React.FC<ModelCategoryRankingsProps> = ({
  initialExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(initialExpanded);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelRankings, setModelRankings] = useState<ModelRankingsData>({});
  const theme = useTheme();

  // Fetch model rankings from the backend
  useEffect(() => {
    const fetchModelRankings = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/api/model-rankings`);

        if (!response.ok) {
          throw new Error(
            `Failed to fetch model rankings: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json();
        setModelRankings(data);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
        setLoading(false);
      }
    };

    fetchModelRankings();
  }, []);

  return (
    <Paper
      elevation={2}
      sx={{
        position: 'relative',
        mb: 2,
        borderRadius: 2,
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        height: { md: expanded ? 'auto' : 'fit-content', xs: 'auto' },
        maxHeight: { md: expanded ? '70vh' : 'auto', xs: 'auto' },
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header - Always visible */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 2,
          bgcolor: theme.palette.grey[50],
          borderBottom: expanded
            ? `1px solid ${theme.palette.divider}`
            : 'none',
          position: 'sticky',
          top: 0,
          zIndex: 1,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <ModelTrainingIcon
            sx={{ mr: 1, color: theme.palette.primary.main }}
          />
          <Typography variant="h6" sx={{ fontWeight: 'medium' }}>
            Model Rankings by Category
          </Typography>
        </Box>
        <IconButton
          onClick={() => setExpanded(!expanded)}
          aria-label={expanded ? 'collapse' : 'expand'}
          sx={{ color: theme.palette.primary.main }}
        >
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      {/* Collapsible Content */}
      <Collapse
        in={expanded}
        timeout="auto"
        sx={{ flexGrow: 1, overflow: 'auto' }}
      >
        <Box
          sx={{
            p: 2,
            bgcolor: 'white',
            overflowY: 'auto',
            maxHeight: { md: 'calc(70vh - 64px)', xs: 'none' }, // Subtract header height
            display: 'flex',
            flexDirection: 'column',
            justifyContent: loading || error ? 'center' : 'flex-start',
            alignItems: loading || error ? 'center' : 'stretch',
            minHeight: loading || error ? '200px' : 'auto',
          }}
        >
          {loading ? (
            <CircularProgress size={40} thickness={4} />
          ) : error ? (
            <Alert
              severity="error"
              icon={<ErrorOutlineIcon />}
              sx={{ width: '100%', maxWidth: '500px' }}
            >
              {error}
            </Alert>
          ) : Object.entries(modelRankings).length === 0 ? (
            <Typography variant="body1" color="text.secondary" align="center">
              No model rankings data available
            </Typography>
          ) : (
            Object.entries(modelRankings).map(([category, data], index) => (
              <Box
                key={category}
                sx={{
                  mb: index < Object.keys(modelRankings).length - 1 ? 3 : 0,
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip
                    label={data.title}
                    sx={{
                      bgcolor: getCategoryColor(category),
                      color: 'white',
                      fontWeight: 'bold',
                      mr: 1,
                    }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {data.description}
                  </Typography>
                </Box>

                <Box sx={{ pl: 1 }}>
                  {data.models.map((model, modelIndex) => (
                    <Box
                      key={model.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        py: 0.5,
                        pl: 1,
                        mb: 0.5,
                        borderLeft: `3px solid ${model.primary ? getCategoryColor(category) : 'transparent'}`,
                        bgcolor: model.primary
                          ? alpha(getCategoryColor(category), 0.05)
                          : 'transparent',
                        borderRadius: '0 4px 4px 0',
                      }}
                    >
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: model.primary ? 'bold' : 'regular',
                          opacity: model.primary ? 1 : 0.8,
                        }}
                      >
                        {model.name}
                      </Typography>
                      {model.primary && (
                        <Chip
                          label="Primary"
                          size="small"
                          variant="outlined"
                          sx={{
                            ml: 1,
                            height: 20,
                            fontSize: '0.7rem',
                            border: `1px solid ${getCategoryColor(category)}`,
                            color: getCategoryColor(category),
                          }}
                        />
                      )}
                    </Box>
                  ))}
                </Box>
              </Box>
            ))
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default ModelCategoryRankings;
