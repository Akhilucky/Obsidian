"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import AnimatedNumber from "@/components/ui/AnimatedNumber"
import { api, withTimeout } from "@/lib/api"
import { fmtPct } from "@/lib/format"

type Holding = {
  symbol: string
  name: string
  qty: number
  avg_cost: number
  last: number
  weight: number
}

const BASE_HOLDINGS: Omit<Holding, "last">[] = [
  { symbol: "AAPL", name: "Apple Inc.", qty: 120, avg_cost: 178.4, weight: 14.2 },
  { symbol: "MSFT", name: "Microsoft", qty: 60, avg_cost: 338.2, weight: 13.8 },
  { symbol: "NVDA", name: "NVIDIA", qty: 95, avg_cost: 89.4, weight: 15.6 },
  { symbol: "GOOGL", name: "Alphabet", qty: 80, avg_cost: 142.1, weight: 9.4 },
  { symbol: "AMZN", name: "Amazon", qty: 55, avg_cost: 158.9, weight: 8.7 },
  { symbol: "META", name: "Meta Platforms", qty: 42, avg_cost: 312.6, weight: 7.9 },
  { symbol: "TSLA", name: "Tesla", qty: 60, avg_cost: 244.3, weight: 6.2 },
  { symbol: "JPM", name: "JPMorgan Chase", qty: 90, avg_cost: 158.7, weight: 5.8 },
  { symbol: "V", name: "Visa", qty: 70, avg_cost: 241.5, weight: 5.1 },
  { symbol: "AMD", name: "AMD", qty: 110, avg_cost: 128.9, weight: 4.3 },
]

const CASH = 125_400

export default function PortfolioPage() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const withPrices = await Promise.all(
        BASE_HOLDINGS.map(async (h) => {
          const q = await withTimeout(api.quote(h.symbol), 15000)
          return { ...h, last: q && q.price > 0 ? q.price : h.avg_cost }
        })
      )
      if (!alive) return
      setHoldings(withPrices)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  const invested = holdings.reduce((s, h) => s + h.qty * h.avg_cost, 0)
  const marketValue = holdings.reduce((s, h) => s + h.qty * h.last, 0)
  const total = marketValue + CASH
  const dayPnl = holdings.reduce((s, h) => s + h.qty * (h.last - h.avg_cost), 0)
  const dayPnlPct = invested > 0 ? (dayPnl / invested) * 100 : 0

  return (
    <Page
      title="Portfolio"
      subtitle="Institutional allocations & risk overview"
      badges={[{ label: "Paper Trading", tone: "live" }]}
    >
      {/* Header metrics */}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <BigMetric label="Total Value" value={total} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Invested" value={invested} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Cash" value={CASH} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Total P&L" value={dayPnl} prefix={dayPnl >= 0 ? "+$" : "-$"} color={dayPnl >= 0 ? "var(--up)" : "var(--down)"} sub={fmtPct(dayPnlPct)} />
      </div>

      {/* Allocation */}
      <div className="mb-6 grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card title="Asset Allocation" className="lg:col-span-1">
          {loading ? (
            <div className="skeleton h-64 w-full" />
          ) : (
            <AllocationPie holdings={holdings} cash={CASH} />
          )}
        </Card>
        <Card title="Sector Exposure" className="lg:col-span-2">
          {loading ? (
            <div className="skeleton h-64 w-full" />
          ) : (
            <div className="space-y-4 pt-2">
              {sectorExposure(holdings).map((s) => (
                <div key={s.name}>
                  <div className="mb-1.5 flex justify-between text-[12px]">
                    <span className="text-[var(--text-muted)]">{s.name}</span>
                    <span className="font-mono text-[var(--text-primary)]">{s.pct.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${s.pct}%`,
                        background: "linear-gradient(90deg, #38bdf8, #2f6fed)",
                        boxShadow: "0 0 10px rgba(56,189,248,0.35)",
                        transition: "width 800ms ease",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Holdings table */}
      <Card title="Holdings" actions={<span className="font-mono text-[10px] uppercase tracking-widest text-[var(--text-muted)]">{holdings.length} positions</span>}>
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-11 w-full" />
            ))}
          </div>
        ) : (
          <table className="obs-table w-full">
            <thead>
              <tr>
                <th>Symbol</th><th>Name</th><th>Qty</th><th>Avg Cost</th><th>Last</th><th>Value</th><th>P&L</th><th>Weight</th>
              </tr>
            </thead>
            <tbody>
              {holdings.map((h) => {
                const value = h.qty * h.last
                const pnl = h.qty * (h.last - h.avg_cost)
                const pnlPct = h.avg_cost > 0 ? ((h.last - h.avg_cost) / h.avg_cost) * 100 : 0
                const pos = pnl >= 0
                return (
                  <tr key={h.symbol}>
                    <td className="font-mono font-semibold text-[var(--text-primary)]">{h.symbol}</td>
                    <td>{h.name}</td>
                    <td className="font-mono">{h.qty}</td>
                    <td className="font-mono">${h.avg_cost.toFixed(2)}</td>
                    <td className="font-mono">${h.last.toFixed(2)}</td>
                    <td className="font-mono text-[var(--text-primary)]">${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                    <td className="font-mono" style={{ color: pos ? "var(--up)" : "var(--down)" }}>
                      {pos ? "+" : ""}${pnl.toLocaleString("en-US", { maximumFractionDigits: 0 })} ({fmtPct(pnlPct)})
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                          <div className="h-full rounded-full" style={{ width: `${h.weight * 4}%`, background: "var(--accent)" }} />
                        </div>
                        <span className="font-mono text-[11px] text-[var(--text-muted)]">{h.weight.toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Card>
    </Page>
  )
}

function BigMetric({ label, value, prefix, color, sub }: { label: string; value: number; prefix: string; color: string; sub?: string }) {
  return (
    <div className="obs-card obs-card-hover p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-1.5 font-mono text-[26px] font-bold" style={{ color }}>
        <AnimatedNumber
          value={Math.abs(value)}
          duration={700}
          format={(v) => `${prefix}${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
        />
      </div>
      {sub && <div className="mt-0.5 font-mono text-[12px]" style={{ color }}>{sub}</div>}
    </div>
  )
}

function AllocationPie({ holdings, cash }: { holdings: Holding[]; cash: number }) {
  const total = holdings.reduce((s, h) => s + h.qty * h.last, 0) + cash
  const segments = [...holdings.map((h) => ({ name: h.symbol, value: (h.qty * h.last / total) * 100 })), { name: "CASH", value: (cash / total) * 100 }]
  let acc = 0
  const COLORS = ["#38bdf8", "#2f6fed", "#7dd3fc", "#818cf8", "#34c88a", "#d9a441", "#a78bfa", "#f472b6", "#22d3ee", "#4ade80", "#f87171"]
  const circles = segments.map((s, i) => {
    const c = (
      <circle
        key={i}
        cx="50" cy="50" r="34" fill="none"
        stroke={COLORS[i % COLORS.length]}
        strokeWidth="12"
        strokeDasharray={`${(s.value / 100) * 213.6} 213.6`}
        strokeDashoffset={-acc * 2.136}
        style={{ transition: "stroke-dasharray 900ms ease" }}
      />
    )
    acc += s.value
    return c
  })
  return (
    <div className="flex flex-col items-center pt-2">
      <div className="relative h-40 w-40">
        <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
          {circles}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-[22px] font-bold text-[var(--text-primary)]">
            {holdings.length}
          </span>
          <span className="text-[9px] uppercase tracking-widest text-[var(--text-muted)]">Positions</span>
        </div>
      </div>
      <div className="mt-3 w-full space-y-1">
        {segments.slice(0, 6).map((s, i) => (
          <div key={s.name} className="flex items-center gap-2 text-[11px]">
            <span className="h-2 w-2 rounded-sm" style={{ background: COLORS[i % COLORS.length] }} />
            <span className="text-[var(--text-muted)]">{s.name}</span>
            <span className="ml-auto font-mono text-[var(--text-primary)]">{s.value.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function sectorExposure(holdings: Holding[]): { name: string; pct: number }[] {
  const tech = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD"]
  const auto = ["TSLA"]
  const fin = ["JPM", "V"]
  const map: Record<string, string[]> = { Technology: tech, "Auto & Mobility": auto, Financials: fin }
  const total = holdings.reduce((s, h) => s + h.qty * h.last, 0)
  const out: { name: string; pct: number }[] = []
  for (const [name, syms] of Object.entries(map)) {
    const value = holdings.filter((h) => syms.includes(h.symbol)).reduce((s, h) => s + h.qty * h.last, 0)
    out.push({ name, pct: total > 0 ? (value / total) * 100 : 0 })
  }
  out.sort((a, b) => b.pct - a.pct)
  return out
}
