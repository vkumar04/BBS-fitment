import { NextResponse } from "next/server";
import { supabase } from "@/app/lib/supabase";

// Simple sentiment analysis (can be improved with a proper library)
function analyzeSentiment(text: string): number {
  const positiveWords = [
    "great",
    "good",
    "awesome",
    "excellent",
    "perfect",
    "thanks",
    "thank you",
    "amazing",
    "love",
  ];
  const negativeWords = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "hate",
    "worst",
    "disappointed",
    "poor",
  ];

  const lowerText = text.toLowerCase();
  let score = 0.5; // neutral

  positiveWords.forEach((word) => {
    if (lowerText.includes(word)) score += 0.05;
  });

  negativeWords.forEach((word) => {
    if (lowerText.includes(word)) score -= 0.05;
  });

  return Math.max(0, Math.min(1, score));
}

export async function POST(req: Request) {
  try {
    const { sessionId, shopDomain, role, content, hasImages } =
      await req.json();

    if (!sessionId || !shopDomain || !role || !content) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 },
      );
    }

    // Get or create conversation
    const { data: existingConversation } = await supabase
      .from("conversations")
      .select("*")
      .eq("session_id", sessionId)
      .single();

    let conversationId: string;

    if (!existingConversation) {
      const { data: newConversation, error } = await supabase
        .from("conversations")
        .insert({
          session_id: sessionId,
          shop_domain: shopDomain,
          is_active: true,
        })
        .select()
        .single();

      if (error || !newConversation) {
        console.error("Error creating conversation:", error);
        return NextResponse.json(
          { error: "Failed to create conversation" },
          { status: 500 },
        );
      }

      conversationId = newConversation.id;
    } else {
      conversationId = existingConversation.id;
    }

    // Analyze sentiment
    const sentiment = analyzeSentiment(content);

    // Insert message
    const { data: message, error: messageError } = await supabase
      .from("messages")
      .insert({
        conversation_id: conversationId,
        role,
        content,
        has_images: hasImages || false,
        sentiment_score: sentiment,
      })
      .select()
      .single();

    if (messageError) {
      console.error("Error creating message:", messageError);
      return NextResponse.json(
        { error: "Failed to create message" },
        { status: 500 },
      );
    }

    // Update conversation stats
    const { data: messages } = await supabase
      .from("messages")
      .select("sentiment_score")
      .eq("conversation_id", conversationId);

    const messageCount = messages?.length || 0;
    const avgSentiment =
      messages && messages.length > 0
        ? messages.reduce((acc, m) => acc + (m.sentiment_score || 0.5), 0) /
          messages.length
        : 0.5;

    await supabase
      .from("conversations")
      .update({
        message_count: messageCount,
        sentiment_score: avgSentiment,
      })
      .eq("id", conversationId);

    return NextResponse.json({
      success: true,
      messageId: message.id,
      conversationId,
    });
  } catch (error) {
    console.error("Error tracking conversation:", error);
    return NextResponse.json(
      { error: "Failed to track conversation" },
      { status: 500 },
    );
  }
}
