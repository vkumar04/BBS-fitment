"use client";

import { useChat } from "@ai-sdk/react";
import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, X, Send, Loader2, User, RotateCcw } from "lucide-react";

// Generate or retrieve session ID
function getSessionId(): string {
  if (typeof window === "undefined") return "";

  let sessionId = sessionStorage.getItem("chatbot_session_id");
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem("chatbot_session_id", sessionId);
  }
  return sessionId;
}

interface FloatingChatbotProps {
  shopDomain?: string;
}

export default function FloatingChatbot({
  shopDomain: propShopDomain = "unknown",
}: FloatingChatbotProps = {}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState(getSessionId);
  const shopDomain = propShopDomain;

  const {
    messages,
    sendMessage: originalSendMessage,
    status,
    stop,
    setMessages,
  } = useChat();

  // Wrap sendMessage to track analytics
  const sendMessage = async (
    message: Parameters<typeof originalSendMessage>[0],
  ) => {
    if (!message) return;

    const content =
      message.parts
        ?.filter((p) => p.type === "text")
        .map((p) => p.text)
        .join(" ") || "";
    const hasImages = message.parts?.some((p) => p.type === "file") || false;

    // Mark that we have a conversation (will be created in track endpoint)
    hasConversationRef.current = true;

    // Track user message
    fetch("/api/analytics/track", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        shopDomain,
        role: "user",
        content,
        hasImages,
      }),
    }).catch((err) => console.error("Track error:", err));

    return originalSendMessage(message);
  };

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const stopRef = useRef(stop);
  const statusRef = useRef(status);
  const hasConversationRef = useRef(false);

  // Keep refs updated
  useEffect(() => {
    stopRef.current = stop;
    statusRef.current = status;
  });

  // Track when a conversation has been created
  useEffect(() => {
    if (messages.length > 0) {
      hasConversationRef.current = true;
    }
  }, [messages.length]);

  // Render text with HTML anchor tags as clickable links
  const renderTextWithLinks = (text: string) => {
    // First, remove all ** markdown characters
    const cleanedText = text.replace(/\*\*/g, "");

    const linkRegex =
      /<a\s+href=['"]([^'"]+)['"](?:\s+target=['"]_blank['"])?>([^<]+)<\/a>/g;
    const parts: (string | React.ReactElement)[] = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(cleanedText)) !== null) {
      if (match.index > lastIndex) {
        parts.push(cleanedText.substring(lastIndex, match.index));
      }

      const url = match[1];
      const linkText = match[2];
      parts.push(
        <a
          key={match.index}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:underline"
        >
          {linkText}
        </a>,
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < cleanedText.length) {
      parts.push(cleanedText.substring(lastIndex));
    }

    return parts.length > 0 ? parts : cleanedText;
  };

  // Communicate with parent window when embedded in iframe
  useEffect(() => {
    if (typeof window !== "undefined" && window.parent !== window) {
      window.parent.postMessage({ type: "chatbot", isOpen }, "*");
    }
  }, [isOpen]);

  // Stop streaming and mark conversation as inactive when chat is closed
  // Mark conversation as active when chat is opened
  useEffect(() => {
    if (!isOpen) {
      // Stop any active streaming
      if (
        statusRef.current === "streaming" ||
        statusRef.current === "submitted"
      ) {
        stopRef.current();
      }

      // Mark conversation as inactive (only if we've created one)
      if (hasConversationRef.current) {
        console.log("Deactivating conversation:", sessionId);
        fetch("/api/analytics/deactivate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId }),
        }).catch((err) => console.error("Deactivate error:", err));
      }
    } else {
      // Mark conversation as active when opened (only if we've created one)
      if (hasConversationRef.current) {
        console.log("Activating conversation:", sessionId);
        fetch("/api/analytics/activate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId }),
        }).catch((err) => console.error("Activate error:", err));
      }
    }
  }, [isOpen, sessionId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Deactivate conversation when page is closed/refreshed
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (hasConversationRef.current) {
        console.log("Page unloading, deactivating conversation:", sessionId);
        // Use fetch with keepalive for reliable delivery during page unload
        fetch("/api/analytics/deactivate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId }),
          keepalive: true,
        }).catch((err) => console.error("Deactivate on unload error:", err));
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [sessionId]);

  // Track assistant responses (only when complete)
  const lastTrackedMessageRef = useRef<string | null>(null);

  useEffect(() => {
    const lastMessage = messages[messages.length - 1];

    if (
      lastMessage &&
      lastMessage.role === "assistant" &&
      status !== "streaming" &&
      lastTrackedMessageRef.current !== lastMessage.id
    ) {
      const content =
        lastMessage.parts
          ?.filter((p) => p.type === "text")
          .map((p) => p.text)
          .join(" ") || "";

      if (content.trim()) {
        lastTrackedMessageRef.current = lastMessage.id;

        fetch("/api/analytics/track", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionId,
            shopDomain,
            role: "assistant",
            content,
            hasImages: false,
          }),
        }).catch((err) => console.error("Track error:", err));
      }
    }
  }, [messages, status, sessionId, shopDomain]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || status === "streaming") {
      return;
    }

    sendMessage({
      role: "user",
      parts: [{ type: "text", text: input }],
    });

    setInput("");
  };

  const handleQuickAction = (vehicle: string) => {
    sendMessage({
      role: "user",
      parts: [{ type: "text", text: vehicle }],
    });
  };

  const handleNewConversation = async () => {
    // Deactivate current conversation
    if (hasConversationRef.current) {
      await fetch("/api/analytics/deactivate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId }),
      }).catch((err) => console.error("Deactivate error:", err));
    }

    // Reset conversation tracking after deactivation
    hasConversationRef.current = false;

    // Clear messages and input
    setMessages([]);
    setInput("");

    // Generate new session ID
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem("chatbot_session_id", newSessionId);
    setSessionId(newSessionId);
  };

  const isLoading = status === "streaming" || status === "submitted";

  const quickPrompts = [
    { label: "What fits my F80?", value: "What fits my F80?" },
    {
      label: "Best setup for F82",
      value: "What's the best setup for F82?",
    },
    {
      label: "M4 GTS options",
      value: "What wheels fit the M4 GTS?",
    },
  ];

  return (
    <>
      {/* Floating Chat Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        size="icon"
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg z-50"
      >
        {isOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <MessageSquare className="h-6 w-6" />
        )}
      </Button>

      {/* Chat Window */}
      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-96 max-w-[calc(100vw-3rem)] h-[750px] max-h-[90vh] flex flex-col z-50 shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b flex-shrink-0">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <h3 className="font-semibold">BBS Fitment Assistant</h3>
            </div>
            <div className="flex items-center gap-1">
              {messages.length > 0 && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleNewConversation}
                  title="New Conversation"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Messages Container */}
          <ScrollArea className="flex-1 min-h-0 p-4 overflow-y-auto">
            {messages.length === 0 && (
              <div className="text-center mt-4">
                <div className="mb-4 flex justify-center">
                  <MessageSquare className="h-16 w-16 text-muted-foreground" />
                </div>
                <p className="text-lg font-semibold mb-2">BBS Fitment Help</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Ask about wheels for your ride
                </p>

                <div className="mt-4">
                  <p className="text-xs text-muted-foreground mb-2">
                    Quick start:
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {quickPrompts.map((prompt) => (
                      <Badge
                        key={prompt.value}
                        variant="secondary"
                        className="cursor-pointer"
                        onClick={() => handleQuickAction(prompt.value)}
                      >
                        {prompt.label}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Disclaimer - only shown on initial screen */}
                <div className="mt-8 px-4 pb-4">
                  <p className="text-[11px] leading-[1.5] text-muted-foreground/70 text-center">
                    All results are for informational purposes only. Fitment
                    must be independently verified. WheelPrice and partners are
                    not liable for any incorrect fitment or resulting costs or
                    damages.
                  </p>
                </div>
              </div>
            )}

            <div className="space-y-4">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {m.role === "assistant" && (
                    <Avatar className="h-8 w-8 shrink-0">
                      <AvatarImage
                        src="/BBS-logo.png"
                        alt="BBS"
                        className="object-contain p-0.5"
                      />
                      <AvatarFallback>BBS</AvatarFallback>
                    </Avatar>
                  )}

                  <div
                    className={`max-w-[75%] rounded-lg px-4 py-3 break-words overflow-hidden ${
                      m.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    {m.parts.map((part, index) => {
                      if (part.type === "text") {
                        return (
                          <div
                            key={index}
                            className="whitespace-pre-wrap text-sm leading-relaxed"
                          >
                            {renderTextWithLinks(part.text)}
                          </div>
                        );
                      }
                      if (
                        part.type === "file" &&
                        part.mediaType?.startsWith("image/")
                      ) {
                        return (
                          <div
                            key={index}
                            className="mt-2 relative w-full max-w-sm"
                          >
                            <Image
                              src={part.url}
                              alt="Uploaded car"
                              width={400}
                              height={300}
                              className="rounded-lg w-full h-auto"
                            />
                          </div>
                        );
                      }
                      return null;
                    })}
                  </div>

                  {m.role === "user" && (
                    <Avatar className="h-8 w-8 shrink-0">
                      <AvatarFallback>
                        <User className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg px-4 py-3 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              )}

              {status === "error" && (
                <div className="flex justify-center">
                  <Badge variant="destructive">
                    Failed to send message. Please try again.
                  </Badge>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input Form */}
          <form
            onSubmit={handleSubmit}
            className="p-4 border-t flex gap-2 flex-shrink-0"
          >
            <Input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="What's your ride?"
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              type="submit"
              size="icon"
              disabled={isLoading || !input.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </Card>
      )}
    </>
  );
}
