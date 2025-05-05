// frontend/app/layout.tsx
// THIS FILE MUST BE A SERVER COMPONENT (NO "use client")

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { CssBaseline } from '@mui/material';
// import { ThemeProvider } from '@mui/material';
// import theme from '../styles/theme'; // Adjust path and uncomment if you have a theme
// import "./globals.css"; // Import global styles if you have them

const inter = Inter({ subsets: ["latin"] });

// Metadata must be exported from a Server Component
export const metadata: Metadata = {
  title: "AI Tools Dashboard", // Customize your title
  description: "Model Selection and Analysis Tools", // Customize description
};

// This is the Root Layout component
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* <ThemeProvider theme={theme}> */}
        <CssBaseline /> {/* MUI base styles */}        
        <body className={inter.className}>
          {children} {/* Your page.tsx content renders here */}
        </body>
      {/* </ThemeProvider> */}
    </html>
  );
}