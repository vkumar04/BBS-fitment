import FloatingChatbot from "../components/FloatingChatbot";

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      {children}
      <FloatingChatbot />
    </>
  );
}
