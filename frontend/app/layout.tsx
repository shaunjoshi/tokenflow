import React from 'react';

export const metadata = {
  title: 'Tokenflow Application',
  description: 'Handles intelligent model selection and prompt processing.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}