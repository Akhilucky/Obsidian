"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import AnimatedNumber from "@/components/ui/AnimatedNumber"
import { api, withTimeout } from "@/lib/api"
import type { ChartPoint } from "@/lib/types"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

const STUDIES = [
  { name: "SMA Crossover", ticker: "AAPL", status: "ready", sharpe: 1.42, winRate: 58.2, nTrades: 132, period: "3Y" },
  { name: "RSI Mean Reversion", ticker: "MSFT", status: "ready", sharpe: 1.18, winRate: 54.7, nTrades: 98, period: "3Y" },
  { name: "MACD Momentum", ticker: "NVDA", status: "ready", sharpe: 1.61, winRate: 61.4, nTrades: 156, period: "3Y" },
  { name: "Breakout", ticker: "GOOGL", status: "running", sharpe: 0.98, winRate: 51.9, nTrades: 87, period: "3Y" },
  { name: "Trend Ensemble", ticker: "AMZN", status: "ready", sharpe: 1.74, winRate: 59.3, nTrades: 145, period: "3Y" },
]

export default function ResearchPage() {
  const [sp500, setSp500] = useState<ChartPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const chart = await withTimeout(api.chart("^GSPC", "1y"), 12000)
      if (!alive) return
      setSp500(chart?.points ?? [])
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  const factors = [
    { name: "Momentum", value: 0.68, color: "#38bdf8" },
    { name: "Value", value: 0.42, color: "#2f6fed" },
    { name: "Quality", value: 0.57, color: "#34c88a" },
    { name: "Low Vol", value: 0.33, color: "#d9a441" },
    { name: "Size", value: 0.21, color: "#a78bfa" },
  ]

  const perfData = sp500.length > 0
    ? sp500.filter((_, i) => i % Math.max(1, Math.floor(sp500.length / 24)) === 0).map((p) => ({ date: p.date.slice(0, 7), sp500: p.close }))
    : []

  return (
    <Page
      title="Research"
      subtitle="Quantitative studies, factors & strategy library"
      badges={[{ label: "C++ Backtest Engine", tone: "up" }]}
    >
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        {/* Backtest summary */}
        <Card title="Strategy Library" className="xl:col-span-2">
          <table className="obs-table w-full">
            <thead>
              <tr><th>Strategy</th><th>Underlying</th><th>Sharpe</th><th>Win Rate</th><th>Trades</th><th>Period</th><th>Status</th></tr>
            </thead>
            <tbody>
              {STUDIES.map((s) => (
                <tr key={s.name}>
                  <td className="font-medium text-[var(--text-primary)]">{s.name}</td>
                  <td className="font-mono">{s.ticker}</td>
                  <td className="font-mono" style={{ color: s.sharpe >= 1.3 ? "var(--up)" : "var(--amber)" }}>
                    {s.sharpe.toFixed(2)}
                  </td>
                  <td className="font-mono">{s.winRate.toFixed(1)}%</td>
                  <td className="font-mono">{s.nTrades}</td>
                  <td className="font-mono">{s.period}</td>
                  <td>
                    <span
                      className="rounded-full border px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider"
                      style={{
                        borderColor: s.status === "ready" ? "rgba(52,200,138,0.35)" : "rgba(56,189,248,0.35)",
                        color: s.status === "ready" ? "var(--up)" : "var(--accent)",
                        background: s.status === "ready" ? "rgba(52,200,138,0.08)" : "rgba(56,189,248,0.08)",
                      }}
                    >
                      {s.status === "ready" ? "● Ready" : "◐ Running"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {/* Factor exposure */}
        <Card title="Factor Exposures">
          <div className="space-y-4 pt-2">
            {factors.map((f, i) => (
              <div key={f.name}>
                <div className="mb-1.5 flex justify-between text-[12px]">
                  <span className="text-[var(--text-muted)]">{f.name}</span>
                  <span className="font-mono text-[var(--text-primary)]">{f.value.toFixed(2)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${f.value * 100}%`,
                      background: f.color,
                      boxShadow: `0 0 10px ${f.color}44`,
                      transition: `width 800ms ease ${i * 60}ms`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 border-t pt-4" style={{ borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between">
              <span className="text-[12px] text-[var(--text-muted)]">Regime</span>
              <span className="rounded-full border px-2.5 py-0.5 font-mono text-[11px]" style={{ borderColor: "rgba(217,164,65,0.35)", color: "var(--amber)" }}>
                Volatile Uptrend
              </span>
            </div>
          </div>
        </Card>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card title="Market Performance" className="lg:col-span-2">
          {loading ? (
            <div className="skeleton h-52 w-full" />
          ) : (
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={perfData} margin={{ top: 8, right: 4, bottom: 0, left: 4 }}>
                  <defs>
                    <linearGradient id="barFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.85} />
                      <stop offset="100%" stopColor="#2f6fed" stopOpacity={0.25} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="2 6" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }} axisLine={false} tickLine={false} width={46} domain={["auto", "auto"]} />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.03)" }}
                    contentStyle={{ background: "rgba(17,17,24,0.92)", border: "1px solid var(--border-strong)", borderRadius: 10, fontSize: 11 }}
                  />
                  <Bar dataKey="sp500" name="S&P 500" fill="url(#barFill)" radius={[3, 3, 0, 0]} animationDuration={800} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Strategy Score">
          <div className="flex flex-col items-center pt-2">
            <div className="relative h-36 w-36">
              <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
                <circle
                  cx="50" cy="50" r="42" fill="none"
                  stroke="url(#scoreGrad)" strokeWidth="10" strokeLinecap="round"
                  strokeDasharray={`${264 * 0.71} 264`}
                  style={{ filter: "drop-shadow(0 0 8px rgba(56,189,248,0.4))", transition: "stroke-dasharray 900ms ease" }}
                />
                <defs>
                  <linearGradient id="scoreGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#38bdf8" />
                    <stop offset="100%" stopColor="#2f6fed" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <AnimatedNumber value={71} duration={900} className="font-mono text-[28px] font-bold text-[var(--text-primary)]" format={(v) => v.toFixed(0)} />
                <span className="text-[9px] uppercase tracking-widest text-[var(--text-muted)]">Overall</span>
              </div>
            </div>
            <div className="mt-3 w-full space-y-1.5">
              <Row label="Signal Quality" value={0.78} color="var(--up)" />
              <Row label="Stability" value={0.64} color="var(--accent)" />
              <Row label="Drawdown Risk" value={0.29} color="var(--down)" />
            </div>
          </div>
        </Card>
      </div>
    </Page>
  )
}

function Row({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center justify-between text-[12px]">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className="font-mono" style={{ color }}>{value.toFixed(2)}</span>
    </div>
  )
}
