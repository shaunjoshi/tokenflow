// lib/theme.ts
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6c47ff', // Vibrant but professional purple
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#b39ddb', // Softer lavender accent
    },
    background: {
      default: '#f4f6f8', // light gray app background
      paper: '#ffffff', // card backgrounds
    },
    text: {
      primary: '#1a1a1a', // very dark gray (better than pure black)
      secondary: '#5f6368', // modern muted gray
    },
    divider: '#e0e0e0',
    error: {
      main: '#e53935',
    },
    warning: {
      main: '#fb8c00',
    },
    info: {
      main: '#42a5f5',
    },
    success: {
      main: '#43a047',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
    },
    subtitle1: {
      fontWeight: 500,
      fontSize: '1rem',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#f4f6f8',
          WebkitFontSmoothing: 'antialiased',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.04)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
        },
        containedPrimary: {
          backgroundColor: '#6c47ff',
          '&:hover': {
            backgroundColor: '#5a3ccc',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 16,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: '#ede7f6', // light purple tint
          color: '#6c47ff',
          fontWeight: 500,
        },
        outlined: {
          borderColor: '#d1b3ff',
          color: '#6c47ff',
        },
      },
    },
    MuiBadge: {
      styleOverrides: {
        badge: {
          backgroundColor: '#e0ccff',
          color: '#6c47ff',
          fontWeight: 500,
        },
      },
    },
  },
});

export default theme;
