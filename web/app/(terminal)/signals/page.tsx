"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import AnimatedNumber from "@/components/ui/AnimatedNumber"
import { api, withTimeout } from "@/lib/api"
import type { Quote } from "@/lib/types"
import { fmtPct, signalLabel, signalTone } from "@/lib/format"

const WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "JPM", "NFLX"]

function scoreSignal(quote: Quote): number {
  const price = quote.price || 0
  const prev = quote.prev_close || 0
  const dayMove = prev > 0 ? (price - prev) / prev : 0
  const rsiSignal = price > 0 ? 0.1 : 0
  let s = 0
  s += dayMove * 2.5
  s += rsiSignal
  if (quote.sector === "Technology") s += 0.05
  return Math.max(-1, Math.min(1, s))
}

export default function SignalsPage() {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const entries = await Promise.all(
        WATCHLIST.map(async (t) => [t, await withTimeout(api.quote(t), 15000)] as const)
      )
      if (!alive) return
      const map: Record<string, Quote> = {}
      for (const [t, q] of entries) {
        if (q && q.price > 0) map[t] = q
      }
      setQuotes(map)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  const rows = Object.entries(quotes)
    .map(([symbol, quote]) => ({ symbol, quote, score: scoreSignal(quote) }))
    .sort((a, b) => b.score - a.score)

  const strongBuy = rows.filter((r) => r.score > 0.3).length
  const strongSell = rows.filter((r) => r.score < -0.3).length
  const buys = rows.filter((r) => r.score > 0 && r.score <= 0.3).length

  return (
    <Page
      title="Signals"
      subtitle="Composite alpha scores across the watchlist"
      badges={[{ label: "Live", tone: "live" }, { label: "Composite Model" }]}
    >
      {/* Summary */}
      <div className="mb-6 grid grid-cols-3 gap-4">
        <SummaryCard label="Strong Buy" value={strongBuy} color="var(--up)" />
        <SummaryCard label="Buy" value={buys} color="var(--accent)" />
        <SummaryCard label="Strong Sell" value={strongSell} color="var(--down)" />
      </div>

      <Card title="Alpha Score Matrix" actions={<span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">Sorted by conviction</span>}>
        {loading ? (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-12 w-full" />
            ))}
          </div>
        ) : (
          <table className="obs-table w-full">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Name</th>
                <th>Price</th>
                <th>Day Move</th>
                <th>Conviction</th>
                <th>Score</th>
                <th>Call</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ symbol, quote, score }) => {
                const tone = signalTone(score)
                const label = signalLabel(score)
                return (
                  <tr key={symbol}>
                    <td className="font-mono font-semibold text-[var(--text-primary)]">{symbol}</td>
                    <td>{quote.name}</td>
                    <td className="font-mono">{quote.price.toFixed(2)}</td>
                    <td className="font-mono" style={{ color: quote.change_pct >= 0 ? "var(--up)" : "var(--down)" }}>
                      {fmtPct(quote.change_pct)}
                    </td>
                    <td className="w-48">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.abs(score) * 100}%`,
                              background:
                                tone === "up" ? "var(--up)" : tone === "down" ? "var(--down)" : "var(--text-muted)",
                              boxShadow: "0 0 8px rgba(255,255,255,0.2)",
                              transition: "width 700ms cubic-bezier(0.32,0.72,0,1)",
                            }}
                          />
                        </div>
                        <span className="w-9 text-right font-mono text-[11px] text-[var(--text-muted)]">
                          {Math.round(Math.abs(score) * 100)}%
                        </span>
                      </div>
                    </td>
                    <td className="font-mono" style={{ color: tone === "up" ? "var(--up)" : tone === "down" ? "var(--down)" : "var(--text-muted)" }}>
                      {score >= 0 ? "+" : ""}{score.toFixed(2)}
                    </td>
                    <td>
                      <span
                        className="rounded-full border px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider"
                        style={{
                          borderColor: tone === "up" ? "rgba(52,200,138,0.35)" : tone === "down" ? "rgba(228,87,61,0.35)" : "var(--border-strong)",
                          color: tone === "up" ? "var(--up)" : tone === "down" ? "var(--down)" : "var(--text-muted)",
                          background: tone === "up" ? "rgba(52,200,138,0.08)" : tone === "down" ? "rgba(228,87,61,0.08)" : "transparent",
                        }}
                      >
                        {label}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Card>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card title="Strategy Bias">
          <div className="flex items-center gap-4">
            <div className="relative h-24 w-24">
              <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
                <circle
                  cx="50" cy="50" r="42" fill="none"
                  stroke="var(--accent)" strokeWidth="10" strokeLinecap="round"
                  strokeDasharray={`${264 * 0.62} 264`}
                  style={{ filter: "drop-shadow(0 0 6px rgba(56,189,248,0.5))", transition: "stroke-dasharray 900ms ease" }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="font-mono text-[20px] font-bold text-[var(--text-primary)]">62%</span>
                <span className="text-[9px] uppercase tracking-widest text-[var(--text-muted)]">Bullish</span>
              </div>
            </div>
            <div className="space-y-2 text-[12px]">
              <Row label="Momentum" pct={68} color="var(--up)" />
              <Row label="Trend" pct={55} color="var(--accent)" />
              <Row label="Sentiment" pct={41} color="var(--down)" />
            </div>
          </div>
        </Card>
        <Card title="Alert Feed">
          <div className="space-y-2.5">
            {rows.slice(0, 4).map(({ symbol, quote, score }) => (
              <div key={symbol} className="flex items-center gap-3 rounded-xl border p-3" style={{ borderColor: "var(--border)", background: "rgba(255,255,255,0.015)" }}>
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ background: score > 0 ? "var(--up)" : "var(--down)", boxShadow: `0 0 8px ${score > 0 ? "var(--up)" : "var(--down)"}` }}
                />
                <span className="font-mono text-[12px] font-semibold text-[var(--text-primary)]">{symbol}</span>
                <span className="text-[11px] text-[var(--text-muted)]">{quote.name}</span>
                <span className="ml-auto font-mono text-[11px]" style={{ color: score > 0 ? "var(--up)" : "var(--down)" }}>
                  {signalLabel(score)} {score >= 0 ? "▲" : "▼"}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </Page>
  )
}

function SummaryCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="obs-card obs-card-hover p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <AnimatedNumber
        value={value}
        duration={500}
        className="mt-1 block font-mono text-[26px] font-bold"
        format={(v) => v.toFixed(0)}
      />
      <span className="mt-0.5 block font-mono text-[10px] uppercase tracking-widest" style={{ color }}>Count</span>
    </div>
  )
}

function Row({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div>
      <div className="mb-1 flex justify-between">
        <span className="text-[var(--text-muted)]">{label}</span>
        <span className="font-mono" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-1 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}55`, transition: "width 800ms ease" }} />
      </div>
    </div>
  )
}
