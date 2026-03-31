import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Human-Scene Interaction System",
  description: "4D human-scene reconstruction and contact estimation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 antialiased">{children}</body>
    </html>
  );
}
