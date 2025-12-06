"use client";

import Image from "next/image";
import { useState } from "react";

export default function Home() {
  const [copied, setCopied] = useState(false);
  const embedCode =
    '<script src="https://bbs-fitment.vercel.app/chatbot-embed.js"></script>';

  const handleCopy = () => {
    navigator.clipboard.writeText(embedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={100}
          height={20}
          priority
        />
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
            BBS Fitment Bot
          </h1>
          <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            AI-powered fitment assistant for BBS wheels. Embed the chatbot on
            your website to help customers find the perfect wheels for their
            vehicle.
          </p>

          {/* How to Embed Section */}
          <div className="w-full mt-8">
            <h2 className="text-xl font-semibold text-black dark:text-zinc-50 mb-4">
              How to Embed
            </h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              Add this script to your website (Shopify, WordPress, or any HTML
              page):
            </p>
            <div className="relative">
              <pre className="bg-zinc-100 dark:bg-zinc-900 p-4 rounded-lg overflow-x-auto text-sm border border-zinc-200 dark:border-zinc-800">
                <code className="text-zinc-800 dark:text-zinc-200">
                  {embedCode}
                </code>
              </pre>
              <button
                onClick={handleCopy}
                className="absolute top-2 right-2 px-3 py-1 text-xs bg-black dark:bg-white text-white dark:text-black rounded hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-colors"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-500 mt-3">
              For Shopify: Add to{" "}
              <code className="bg-zinc-100 dark:bg-zinc-900 px-1 py-0.5 rounded">
                layout/theme.liquid
              </code>{" "}
              before the closing{" "}
              <code className="bg-zinc-100 dark:bg-zinc-900 px-1 py-0.5 rounded">
                &lt;/body&gt;
              </code>{" "}
              tag
            </p>
          </div>
        </div>
        <div className="flex flex-col gap-4 text-base font-medium sm:flex-row">
          <a
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-foreground px-5 text-background transition-colors hover:bg-[#383838] dark:hover:bg-[#ccc] md:w-[158px]"
            href="/dashboard"
          >
            Dashboard
          </a>
        </div>
      </main>
    </div>
  );
}
