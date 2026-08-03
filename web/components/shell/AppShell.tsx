"use client"

import { useEffect, useState } from "react"
import { usePathname } from "next/navigation"
import { AnimatePresence } from "framer-motion"
import Sidebar from "./Sidebar"
import Topbar from "./Topbar"
import CommandPalette from "./CommandPalette"
import type { ReactNode } from "react"

export default function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const [commandOpen, setCommandOpen] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setCommandOpen((v) => !v)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const marginLeft = collapsed ? 80 : 264

  return (
    <div className="grain min-h-screen">
      <Sidebar collapsed={collapsed} />
      <Topbar
        collapsed={collapsed}
        onToggle={() => setCollapsed((v) => !v)}
        onCommand={() => setCommandOpen(true)}
      />
      <main
        className="px-7 pb-16 pt-[84px]"
        style={{
          marginLeft,
          maxWidth: 1600,
          transition: "margin-left 260ms cubic-bezier(0.32,0.72,0,1)",
        }}
      >
        <AnimatePresence mode="wait" initial={false}>
          <div key={pathname}>{children}</div>
        </AnimatePresence>
      </main>
      <CommandPalette open={commandOpen} onClose={() => setCommandOpen(false)} />
    </div>
  )
}
