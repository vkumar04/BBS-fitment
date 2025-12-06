import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dashboard - BBS Fitment Bot",
  description: "Analytics dashboard for fitment bot",
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
