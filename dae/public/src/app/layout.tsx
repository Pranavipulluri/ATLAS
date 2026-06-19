import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Agentic Edge Traffic Management | God's Eye Dashboard",
  description:
    "Real-time decentralized traffic grid simulation with multi-agent communication and predictive emergency routing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-gray-50 text-gray-900 font-[var(--font-inter)] antialiased">
        {children}
      </body>
    </html>
  );
}
