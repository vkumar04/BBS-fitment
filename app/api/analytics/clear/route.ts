import { NextResponse } from "next/server";
import { supabase } from "@/app/lib/supabase";

export async function POST() {
  try {
    // Delete all data from tables
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

    return NextResponse.json({
      success: true,
      message: "Database cleared successfully",
    });
  } catch (error) {
    console.error("Error clearing database:", error);
    return NextResponse.json(
      { error: "Failed to clear database" },
      { status: 500 },
    );
  }
}
