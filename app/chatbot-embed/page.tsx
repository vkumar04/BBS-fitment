"use client";

import FloatingChatbot from "../components/FloatingChatbot";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

function ChatbotEmbedContent() {
  const searchParams = useSearchParams();
  const shop = searchParams.get("shop") || "unknown";

  return (
    <div>
      <FloatingChatbot shopDomain={shop} />
    </div>
  );
}

export default function ChatbotEmbed() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ChatbotEmbedContent />
    </Suspense>
  );
}
