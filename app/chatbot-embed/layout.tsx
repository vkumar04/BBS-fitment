import type { Metadata } from "next";
import "../globals.css";

export const metadata: Metadata = {
  title: "Chatbot",
  description: "AI Chatbot Embed",
};

export default function ChatbotEmbedLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, background: 'transparent' }}>
        {children}
      </body>
    </html>
  );
}
