"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createClient } from "@/utils/supabase/server";

// Feature flag to enable/disable signup - set to false to disable new user signups
const ENABLE_SIGNUP = true;

export async function sendMagicLink(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;

  // If signup is disabled, check if user exists first
  if (!ENABLE_SIGNUP) {
    const { data: users, error: checkError } = await supabase
      .from("auth.users")
      .select("email")
      .eq("email", email)
      .single();

    // If user doesn't exist and signup is disabled, throw error
    if (checkError || !users) {
      throw new Error("Account not found. Please contact your administrator.");
    }
  }

  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo: `${process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"}/auth/callback`,
      shouldCreateUser: ENABLE_SIGNUP,
    },
  });

  if (error) {
    throw new Error(error.message || "Failed to send magic link");
  }

  revalidatePath("/", "layout");
}

export async function signout() {
  const supabase = await createClient();

  const { error } = await supabase.auth.signOut();

  if (error) {
    throw new Error("Failed to sign out");
  }

  revalidatePath("/", "layout");
  redirect("/login");
}
