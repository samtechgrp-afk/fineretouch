import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FineRetouch',
  description: 'AI-assisted batch portrait retouching for high-volume photographers.',
};

export const viewport: Viewport = { themeColor: '#0d0e10' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
