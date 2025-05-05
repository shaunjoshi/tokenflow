// src/Dashboard.tsx - Refactored for Navigation
import React, {useCallback, useState} from 'react';
import axios, {AxiosInstance} from 'axios';
import {supabase} from '../clients/supabaseClient'; // Adjust path if needed
import {Session} from '@supabase/supabase-js';
import ModelSelectionChatView from './ModelSelectionChatView'; // Import model selection view
import {
    AppBar,
    Box,
    CssBaseline,
    Divider,
    Drawer,
    IconButton,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Stack,
    Toolbar,
    Typography,
    useTheme
} from '@mui/material';
import LogoutIcon from '@mui/icons-material/Logout';
import InsightsIcon from '@mui/icons-material/Insights'; // Sentiment Icon
import MenuIcon from '@mui/icons-material/Menu'; // For potential mobile drawer toggle
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'; // For Model Selection

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
const drawerWidth = 240; // Define drawer width

// --- Define Feature Key Type ---
type FeatureKey = 'sentiment' | 'ranker' | 'model-selector';

interface DashboardProps {
    session: Session;
}

function Dashboard({ session }: DashboardProps) {
    // Log that Dashboard function body is executing
    console.log("--- Dashboard Component Render Start ---"); 

    const [selectedFeature, setSelectedFeature] = useState<FeatureKey>('model-selector'); // Default to model selector
    const [mobileOpen, setMobileOpen] = useState(false); // State for mobile drawer
    const [isClosing, setIsClosing] = useState(false); // Prevent flicker on mobile close
    const theme = useTheme();

    // --- API Client Factory (Stays Here or move to context/util) ---
    const apiClient = useCallback((): AxiosInstance | null => {
        if (!session?.access_token) { console.error("No session token"); return null; }
        if (!API_BASE_URL) { console.error("API Base URL missing"); return null; }
        return axios.create({
            baseURL: API_BASE_URL,
            headers: { Authorization: `Bearer ${session.access_token}`, 'Content-Type': 'application/json' },
            timeout: 45000,
        });
    }, [session]);

    // --- Event Handlers ---
    const handleLogout = useCallback(async () => {
        await supabase.auth.signOut();
        // App.js listener will redirect
    }, []);

    const handleFeatureSelect = (feature: FeatureKey) => {
        setSelectedFeature(feature);
        if(mobileOpen) handleDrawerToggle(); // Close mobile drawer on selection
    };

    const handleDrawerToggle = () => {
        if (!isClosing) {
            setMobileOpen(!mobileOpen);
        }
    };

    const handleDrawerClose = () => {
        setIsClosing(true);
        setMobileOpen(false);
    };

    const handleDrawerTransitionEnd = () => {
        setIsClosing(false);
    };

    // --- Get current feature title ---
    const getCurrentFeatureTitle = () => {
        switch(selectedFeature) {
            case 'sentiment': return 'Sentiment Analysis';
            case 'ranker': return 'Product Ranker';
            case 'model-selector': return 'Model Selection Chat';
            default: return 'Dashboard';
        }
    };

    // --- Drawer Content ---
    const drawerContent = (
        <div>
            <Toolbar>
                {/* Optional: Logo/Title in Drawer Header */}
                <Stack direction="row" alignItems="center" spacing={1}>
                    <InsightsIcon color="primary" />
                    <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 'bold'}}>
                        AI Tools
                    </Typography>
                </Stack>
            </Toolbar>
            <Divider />
            <List>
                <ListItem disablePadding>
                    <ListItemButton
                        selected={selectedFeature === 'model-selector'}
                        onClick={() => handleFeatureSelect('model-selector')}
                    >
                        <ListItemIcon>
                            <AutoAwesomeIcon color={selectedFeature === 'model-selector' ? 'primary' : 'action'} />
                        </ListItemIcon>
                        <ListItemText primary="Model Selection Chat" />
                    </ListItemButton>
                </ListItem>
                {/* Add more features here */}
            </List>
            <Divider sx={{ mt: 'auto', mb: 1 }} /> {/* Push logout towards bottom */}
            <List>
                <ListItem disablePadding>
                    <ListItemButton onClick={handleLogout} sx={{ color: 'error.main' }}>
                        <ListItemIcon sx={{ color: 'error.main' }}>
                            <LogoutIcon />
                        </ListItemIcon>
                        <ListItemText primary="Logout" />
                    </ListItemButton>
                </ListItem>
            </List>
        </div>
    );

    // --- Render ---
    // Log the state right before rendering conditional views
    console.log("Dashboard rendering feature:", selectedFeature);

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <CssBaseline /> {/* Ensures consistent baseline */}
            {/* Header (AppBar) */}
            <AppBar
                position="fixed" // Fixed position
                sx={{
                    // Calculate width to exclude drawer space on larger screens
                    width: { sm: `calc(100% - ${drawerWidth}px)` },
                    ml: { sm: `${drawerWidth}px` }, // Margin left for permanent drawer space
                    backgroundColor: 'background.paper',
                    color: 'text.primary',
                    zIndex: theme.zIndex.drawer + 1 // Ensure AppBar is above drawer
                }}
            >
                <Toolbar>
                    {/* Mobile Menu Button */}
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' } }} // Only show on small screens
                    >
                        <MenuIcon />
                    </IconButton>
                    {/* Current Feature Title */}
                    <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                        {getCurrentFeatureTitle()}
                    </Typography>
                    {/* User Email could go here again if desired */}
                    {/* Logout button is now in the Drawer */}
                </Toolbar>
            </AppBar>

            {/* Navigation Drawer */}
            <Box
                component="nav"
                sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
                aria-label="mailbox folders"
            >
                {/* Temporary Drawer for Mobile */}
                <Drawer
                    variant="temporary"
                    open={mobileOpen}
                    onTransitionEnd={handleDrawerTransitionEnd}
                    onClose={handleDrawerClose}
                    ModalProps={{
                        keepMounted: true, // Better open performance on mobile.
                    }}
                    sx={{
                        display: { xs: 'block', sm: 'none' }, // Show only on xs
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                >
                    {drawerContent}
                </Drawer>
                {/* Permanent Drawer for Desktop */}
                <Drawer
                    variant="permanent"
                    sx={{
                        display: { xs: 'none', sm: 'block' }, // Hide on xs
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                    open // Permanent is always open on desktop
                >
                    {drawerContent}
                </Drawer>
            </Box>

            {/* Main Content Area */}
            <Box
                component="main"
                sx={{
                    flexGrow: 1, // Takes remaining space
                    p: { xs: 2, sm: 3 }, // Responsive padding
                    width: { sm: `calc(100% - ${drawerWidth}px)` }, // Adjust width for drawer
                    mt: '64px', // Offset for AppBar height (adjust if your AppBar height differs)
                    backgroundColor: (theme) => theme.palette.grey[100] // Slight background differentiation
                }}
            >
                {/* Conditionally render the selected feature's view */}
                {selectedFeature === 'model-selector' && (
                    <ModelSelectionChatView session={session} apiClient={apiClient} />
                )}
                {/* Add more feature views here based on selectedFeature */}
            </Box>
        </Box>
    );
}

export default Dashboard;