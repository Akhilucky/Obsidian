"use client"

import AppShell from "@/components/shell/AppShell";
import type { ReactNode } from "react";

export default function TerminalLayout({ children }: { children: ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
