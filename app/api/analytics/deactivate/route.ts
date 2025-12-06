import { NextResponse } from "next/server";
import { supabase } from "@/app/lib/supabase";

export async function POST(req: Request) {
  try {
    const { sessionId } = await req.json();

    if (!sessionId) {
      return NextResponse.json({ error: "Missing sessionId" }, { status: 400 });
    }

    // Mark conversation as inactive
    const { data, error } = await supabase
      .from("conversations")
      .update({ is_active: false })
      .eq("session_id", sessionId)
      .select();

    if (error) {
      console.error("Error deactivating conversation:", error);
      return NextResponse.json(
        { error: "Failed to deactivate conversation" },
        { status: 500 },
      );
    }

    console.log(`Deactivated conversation for session ${sessionId}:`, data);
    return NextResponse.json({ success: true, updated: data?.length || 0 });
  } catch (error) {
    console.error("Error in deactivate endpoint:", error);
    return NextResponse.json(
      { error: "Failed to deactivate conversation" },
      { status: 500 },
    );
  }
}
