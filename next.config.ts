import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        // Allow embedding in iframes from any Shopify store
        source: "/chatbot-embed",
        headers: [
          {
            key: "Access-Control-Allow-Origin",
            value: "*", // In production, replace with your specific Shopify domain
          },
          {
            key: "X-Frame-Options",
            value: "ALLOWALL",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
