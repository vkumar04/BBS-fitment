"use client";

import {
  useActionState,
  useOptimistic,
  useCallback,
  useTransition,
} from "react";
import { useEffect, useState } from "react";
import Image from "next/image";
import { clearDatabase, getMetrics } from "./actions";
import { signout } from "../login/actions";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { MessageSquare, TrendingUp, Users, BarChart3 } from "lucide-react";

interface Metrics {
  totalConversations: number;
  todayConversations: number;
  activeConversations: number;
  averageSentiment: number;
  shopBreakdown: Array<{ shop_domain: string; session_count: number }>;
  recentConversations: Array<{
    id: string;
    session_id: string;
    shop_domain: string;
    started_at: string;
    message_count: number;
    sentiment_score: number;
    is_active: number;
  }>;
}

async function clearDatabaseWithConfirm() {
  if (
    typeof window !== "undefined" &&
    !confirm("Are you sure you want to clear all data? This cannot be undone.")
  ) {
    return { success: false, error: "Cancelled by user" };
  }
  return clearDatabase();
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [, startTransition] = useTransition();

  // Use useActionState for the clear database action (React 19)
  const [clearState, clearAction, isPending] = useActionState(async () => {
    return await clearDatabaseWithConfirm();
  }, null);

  // Use optimistic updates
  const [optimisticMetrics, setOptimisticMetrics] = useOptimistic(
    metrics,
    (state, newMetrics: Metrics) => newMetrics,
  );

  // Use useCallback to memoize the fetch function
  const fetchMetrics = useCallback(async () => {
    try {
      const data = await getMetrics();
      setMetrics(data);
      // Wrap optimistic update in startTransition
      startTransition(() => {
        setOptimisticMetrics(data);
      });
    } catch (error) {
      console.error("Error fetching metrics:", error);
      toast.error("Failed to fetch metrics");
    } finally {
      setLoading(false);
    }
  }, [setOptimisticMetrics, startTransition]);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(() => {
      fetchMetrics();
    }, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  // Show success message after clearing and refresh metrics
  useEffect(() => {
    if (clearState?.success) {
      toast.success("Database cleared successfully");
      fetchMetrics();
    } else if (clearState?.error && clearState.error !== "Cancelled by user") {
      toast.error("Failed to clear database");
    }
  }, [clearState, fetchMetrics]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  const displayMetrics = optimisticMetrics || metrics;

  // Calculate percentage changes
  const totalChange = displayMetrics?.totalConversations
    ? (
        (displayMetrics.todayConversations /
          Math.max(
            displayMetrics.totalConversations -
              displayMetrics.todayConversations,
            1,
          )) *
        100
      ).toFixed(1)
    : "0.0";

  const todayChange = displayMetrics?.todayConversations
    ? (
        (displayMetrics.activeConversations /
          Math.max(displayMetrics.todayConversations, 1)) *
        100
      ).toFixed(1)
    : "0.0";

  const sentimentChange = (
    (displayMetrics?.averageSentiment || 0.5) - 0.5
  ).toFixed(2);

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Image
            src="/BBS-logo.png"
            alt="BBS"
            width={120}
            height={40}
            className="object-contain"
          />
        </div>
        <div className="flex gap-3">
          <form action={signout}>
            <Button type="submit" variant="outline">
              Sign Out
            </Button>
          </form>
          <form action={clearAction}>
            <Button type="submit" variant="destructive" disabled={isPending}>
              {isPending ? "Clearing..." : "Clear Database"}
            </Button>
          </form>
        </div>
      </div>

      {/* Overview Section */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Overview</h1>
        <p className="text-muted-foreground">
          Monitor fitment assistant performance and customer interactions
        </p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Total Conversations */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Conversations
            </CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {displayMetrics?.totalConversations || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              <span
                className={
                  parseFloat(totalChange) >= 0
                    ? "text-green-500"
                    : "text-red-500"
                }
              >
                {parseFloat(totalChange) >= 0 ? "+" : ""}
                {totalChange}%
              </span>{" "}
              from last period
            </p>
          </CardContent>
        </Card>

        {/* Today */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Today</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {displayMetrics?.todayConversations || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              <span
                className={
                  parseFloat(todayChange) >= 0
                    ? "text-green-500"
                    : "text-red-500"
                }
              >
                {parseFloat(todayChange) >= 0 ? "+" : ""}
                {todayChange}%
              </span>{" "}
              active now
            </p>
          </CardContent>
        </Card>

        {/* Active Now */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Now</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {displayMetrics?.activeConversations || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Live conversations
            </p>
          </CardContent>
        </Card>

        {/* Avg Sentiment */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Avg. Sentiment
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(displayMetrics?.averageSentiment || 0.5).toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              <span
                className={
                  parseFloat(sentimentChange) >= 0
                    ? "text-green-500"
                    : "text-red-500"
                }
              >
                {parseFloat(sentimentChange) >= 0 ? "+" : ""}
                {sentimentChange}
              </span>{" "}
              from neutral
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Conversations */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Conversations</CardTitle>
          <CardDescription>
            Latest customer interactions with the fitment assistant
          </CardDescription>
        </CardHeader>
        <CardContent>
          {displayMetrics?.recentConversations &&
          displayMetrics.recentConversations.length > 0 ? (
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-4">
                {displayMetrics.recentConversations.map((conv, index) => (
                  <div key={conv.id}>
                    <div className="flex items-center justify-between py-4">
                      <div className="flex items-center gap-4">
                        <div
                          className={`w-2 h-2 rounded-full ${conv.is_active ? "bg-green-500" : "bg-gray-500"}`}
                        />
                        <div>
                          <div className="font-medium">
                            Conversation #
                            {displayMetrics.recentConversations.length - index}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {new Date(conv.started_at).toLocaleString()} •{" "}
                            {conv.message_count} messages
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground mb-1">
                          Sentiment
                        </div>
                        <Badge
                          variant={
                            conv.sentiment_score > 0.6
                              ? "default"
                              : conv.sentiment_score < 0.4
                                ? "destructive"
                                : "secondary"
                          }
                        >
                          {conv.sentiment_score.toFixed(2)}
                        </Badge>
                      </div>
                    </div>
                    {index < displayMetrics.recentConversations.length - 1 && (
                      <Separator />
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              No conversations yet
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
