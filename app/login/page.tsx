"use client";

import { useState } from "react";
import { sendMagicLink } from "./actions";
import Image from "next/image";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

// Feature flag to enable/disable signup - set to false to disable new user signups
const ENABLE_SIGNUP = true;

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  const handleMagicLink = async (formData: FormData) => {
    setIsLoading(true);
    try {
      await sendMagicLink(formData);
      setEmailSent(true);
      toast.success("Check your email for the magic link!");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to send magic link";
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <Image
            src="/BBS-logo.png"
            alt="BBS"
            width={150}
            height={50}
            className="object-contain"
          />
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Welcome</CardTitle>
            <CardDescription>
              {emailSent
                ? "We've sent you a magic link! Check your email to sign in."
                : "Enter your email to receive a magic link"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!emailSent ? (
              <form action={handleMagicLink} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="you@example.com"
                    required
                    disabled={isLoading}
                  />
                </div>

                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? "Sending magic link..." : "Send Magic Link"}
                </Button>
              </form>
            ) : (
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Click the link in your email to sign in. You can close this
                  page.
                </p>
                <Button
                  onClick={() => setEmailSent(false)}
                  variant="outline"
                  className="w-full"
                >
                  Send Another Link
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {!ENABLE_SIGNUP && (
          <p className="text-muted-foreground text-sm text-center mt-4">
            New signups are disabled. Contact your administrator for access.
          </p>
        )}
      </div>
    </div>
  );
}
