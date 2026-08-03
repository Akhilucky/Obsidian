"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion } from "framer-motion"
import {
  Activity,
  BarChart3,
  Bot,
  FlaskConical,
  LayoutDashboard,
  Landmark,
  Radar,
  Settings,
  Wallet,
} from "lucide-react"

const NAV = [
  { section: "Markets", items: [
    { label: "Overview", path: "/overview", icon: LayoutDashboard },
    { label: "Analysis", path: "/analysis", icon: BarChart3 },
    { label: "Signals", path: "/signals", icon: Radar },
    { label: "India", path: "/india", icon: Landmark },
  ]},
  { section: "Management", items: [
    { label: "Portfolio", path: "/portfolio", icon: Wallet },
    { label: "Research", path: "/research", icon: FlaskConical },
    { label: "Agents", path: "/agents", icon: Bot },
  ]},
  { section: "System", items: [
    { label: "Settings", path: "/settings", icon: Settings },
  ]},
]

export default function Sidebar({ collapsed }: { collapsed: boolean }) {
  const pathname = usePathname()
  const width = collapsed ? 80 : 264

  return (
    <motion.aside
      initial={false}
      animate={{ width }}
      transition={{ duration: 0.26, ease: [0.32, 0.72, 0, 1] }}
      className="fixed left-0 top-0 z-40 flex h-screen flex-col border-r backdrop-blur-xl"
      style={{ borderColor: "var(--border)", background: "rgba(9,9,11,0.82)" }}
    >
      {/* Brand */}
      <div className="flex h-[64px] items-center px-5">
        <motion.div
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
          style={{
            background: "linear-gradient(135deg,#38bdf8,#2f6fed)",
            boxShadow: "0 0 24px rgba(56,189,248,0.35)",
          }}
        >
          <Activity size={18} strokeWidth={2.2} color="#08121a" />
        </motion.div>
        <motion.div
          animate={{ opacity: collapsed ? 0 : 1, x: collapsed ? -6 : 0 }}
          transition={{ duration: 0.18 }}
          className="ml-3 overflow-hidden"
        >
          <div className="text-[15px] font-bold tracking-wide text-[var(--text-primary)]">
            Obsidian
          </div>
          <div className="font-mono text-[9px] uppercase tracking-[0.22em] accent-gradient">
            Terminal
          </div>
        </motion.div>
      </div>

      {/* Nav */}
      <div className="flex-1 overflow-y-auto px-3 pb-6 pt-2">
        {NAV.map((group) => (
          <div key={group.section} className="mb-5">
            <motion.div
              animate={{ opacity: collapsed ? 0 : 1, height: collapsed ? 0 : 18 }}
              className="px-3 text-[9px] font-semibold uppercase tracking-[0.2em] text-[var(--text-muted)]"
            >
              {group.section}
            </motion.div>
            <div className="flex flex-col gap-0.5">
              {group.items.map(({ label, path, icon: Icon }) => {
                const active = pathname === path
                return (
                  <Link key={path} href={path}>
                    <div
                      className="relative flex h-10 items-center rounded-xl transition-colors duration-150"
                      style={{
                        background: active
                          ? "linear-gradient(90deg, rgba(56,189,248,0.12), transparent)"
                          : "transparent",
                      }}
                    >
                      {active && (
                        <motion.div
                          layoutId="nav-indicator"
                          className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full"
                          style={{ background: "var(--accent)", boxShadow: "0 0 12px var(--accent-glow)" }}
                          transition={{ type: "spring", stiffness: 500, damping: 40 }}
                        />
                      )}
                      <div
                        className="mx-3 flex items-center gap-3"
                        style={{ paddingLeft: collapsed ? 2 : 0 }}
                      >
                        <Icon
                          size={18}
                          strokeWidth={1.8}
                          style={{
                            color: active ? "var(--accent)" : "var(--text-muted)",
                            transition: "color 150ms",
                          }}
                        />
                        {!collapsed && (
                          <span
                            className="text-[13px] whitespace-nowrap"
                            style={{
                              color: active ? "var(--text-primary)" : "var(--text-secondary)",
                            }}
                          >
                            {label}
                          </span>
                        )}
                      </div>
                    </div>
                  </Link>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Footer status */}
      <div className="border-t px-4 py-4" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2.5">
          <span
            className="live-dot h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ background: "var(--up)", boxShadow: "0 0 8px var(--up)" }}
          />
          {!collapsed && (
            <div className="overflow-hidden">
              <div className="text-[11px] font-medium text-[var(--text-secondary)]">
                Pipeline Active
              </div>
              <div className="font-mono text-[9px] uppercase tracking-widest text-[var(--text-muted)]">
                Obsidian Core v1
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  )
}
