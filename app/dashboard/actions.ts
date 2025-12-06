"use server";

import { supabase } from "@/app/lib/supabase";
import { revalidatePath } from "next/cache";

export async function clearDatabase() {
  try {
    // Delete all data from tables (Supabase doesn't support DELETE without WHERE, so we use a condition that matches all)
    await supabase
      .from("messages")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000");
    await supabase
      .from("conversations")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000");
    await supabase
      .from("daily_metrics")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000");

    // Revalidate the dashboard to refresh data
    revalidatePath("/dashboard");

    return { success: true, message: "Database cleared successfully" };
  } catch (error) {
    console.error("Error clearing database:", error);
    return { success: false, error: "Failed to clear database" };
  }
}

export async function getMetrics() {
  try {
    // Get total conversations
    const { count: totalConversations } = await supabase
      .from("conversations")
      .select("*", { count: "exact", head: true });

    // Get today's conversations
    const today = new Date().toISOString().split("T")[0];
    const { count: todayConversations } = await supabase
      .from("conversations")
      .select("*", { count: "exact", head: true })
      .gte("started_at", `${today}T00:00:00`)
      .lt("started_at", `${today}T23:59:59`);

    // Get active conversations
    const { count: activeConversations } = await supabase
      .from("conversations")
      .select("*", { count: "exact", head: true })
      .eq("is_active", true);

    // Get average sentiment
    const { data: sentimentData } = await supabase
      .from("conversations")
      .select("sentiment_score")
      .not("sentiment_score", "is", null);

    const averageSentiment =
      sentimentData && sentimentData.length > 0
        ? sentimentData.reduce((acc, row) => acc + row.sentiment_score, 0) /
          sentimentData.length
        : 0.5;

    // Get shop breakdown
    const { data: shopBreakdown } = await supabase
      .from("conversations")
      .select("shop_domain")
      .then(({ data }) => {
        if (!data) return { data: [] };
        const grouped = data.reduce((acc: Record<string, number>, row) => {
          acc[row.shop_domain] = (acc[row.shop_domain] || 0) + 1;
          return acc;
        }, {});
        return {
          data: Object.entries(grouped)
            .map(([shop_domain, session_count]) => ({
              shop_domain,
              session_count: session_count as number,
            }))
            .sort((a, b) => b.session_count - a.session_count),
        };
      });

    // Get recent conversations with details
    const { data: recentConversations } = await supabase
      .from("conversations")
      .select(
        "id, session_id, shop_domain, started_at, message_count, sentiment_score, is_active",
      )
      .order("started_at", { ascending: false })
      .limit(10);

    return {
      totalConversations: totalConversations || 0,
      todayConversations: todayConversations || 0,
      activeConversations: activeConversations || 0,
      averageSentiment,
      shopBreakdown: shopBreakdown || [],
      recentConversations: (recentConversations || []).map((conv) => ({
        ...conv,
        is_active: conv.is_active ? 1 : 0,
      })),
    };
  } catch (error) {
    console.error("Error fetching metrics:", error);
    throw new Error("Failed to fetch metrics");
  }
}
