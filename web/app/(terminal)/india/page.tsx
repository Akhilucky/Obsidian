"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import TickerStrip from "@/components/ui/TickerStrip"
import Card from "@/components/ui/Card"
import MetricCard from "@/components/ui/MetricCard"
import PriceChart from "@/components/charts/PriceChart"
import { api, withTimeout } from "@/lib/api"
import type { ChartPoint, IndexQuote, IndiaPopular, Quote } from "@/lib/types"
import { fmtPct } from "@/lib/format"

const POPULAR = [
  "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
  "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
]

export default function IndiaPage() {
  const [indices, setIndices] = useState<Record<string, IndexQuote> | null>(null)
  const [ticker, setTicker] = useState("RELIANCE.NS")
  const [quote, setQuote] = useState<Quote | null>(null)
  const [points, setPoints] = useState<ChartPoint[]>([])
  const [popular, setPopular] = useState<IndiaPopular[]>([])
  const [loading, setLoading] = useState(true)

  const selectTicker = (t: string) => {
    setTicker(t)
    setLoading(true)
  }

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [idx, india] = await Promise.all([
        withTimeout(api.indices("india"), 12000),
        withTimeout(api.india(ticker, "1y"), 12000),
      ])
      if (!alive) return
      setIndices(idx)
      if (india) {
        setQuote(india.quote)
        setPoints(india.points)
        setPopular(india.popular)
      }
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [ticker])

  const up = (quote?.change_pct ?? 0) >= 0

  return (
    <Page
      title="Indian Markets"
      subtitle="NSE / BSE real-time data & analysis"
      badges={[{ label: "NSE Live", tone: "live" }]}
    >
      <TickerStrip items={indices ?? {}} loading={!indices} />

      <div className="mb-6 flex flex-wrap items-center gap-2">
        {POPULAR.map((t) => (
          <button
            key={t}
            onClick={() => selectTicker(t)}
            className="rounded-lg border px-3 py-1.5 font-mono text-[12px] transition-all duration-150 active:scale-95"
            style={{
              borderColor: t === ticker ? "var(--accent)" : "var(--border-strong)",
              background: t === ticker ? "rgba(56,189,248,0.10)" : "rgba(255,255,255,0.02)",
              color: t === ticker ? "var(--accent)" : "var(--text-secondary)",
              boxShadow: t === ticker ? "0 0 16px rgba(56,189,248,0.15)" : "none",
            }}
          >
            {t.replace(".NS", "")}
          </button>
        ))}
      </div>

      <Card
        title={`${ticker} — 1 Year`}
        badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>NSE</span>}
        className="obs-card-hover"
      >
        {loading ? (
          <div className="skeleton h-[340px] w-full" />
        ) : (
          <>
            <div className="mb-3 flex items-baseline gap-3">
              <span className="font-mono text-[30px] font-bold text-[var(--text-primary)]">
                ₹{(quote?.price ?? 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span className="font-mono text-[14px]" style={{ color: up ? "var(--up)" : "var(--down)" }}>
                {quote ? fmtPct(quote.change_pct) : ""}
              </span>
              <span className="text-[11px] text-[var(--text-muted)]">{quote?.name}</span>
            </div>
            <PriceChart points={points} height={340} />
          </>
        )}
      </Card>

      <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard label="Open" value={quote?.open ?? 0} prefix="₹" />
        <MetricCard label="Day High" value={quote?.high ?? 0} prefix="₹" />
        <MetricCard label="Day Low" value={quote?.low ?? 0} prefix="₹" />
        <MetricCard label="Volume" value={quote?.volume ?? 0} prefix="" />
      </div>

      <Card title="Popular Indian Stocks" className="mt-5">
        {loading && popular.length === 0 ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="skeleton h-20 w-full" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            {popular.map((p, i) => {
              const pos = (p.change_pct ?? 0) >= 0
              return (
                <button
                  key={p.symbol}
                  onClick={() => selectTicker(p.symbol)}
                  className="obs-card obs-card-hover p-3.5 text-left"
                  style={{ animationDelay: `${i * 40}ms` }}
                >
                  <div className="font-mono text-[13px] font-semibold text-[var(--text-primary)]">
                    {p.symbol.replace(".NS", "")}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">{p.name}</div>
                  <div className="mt-1.5 font-mono text-[13px] text-[var(--text-primary)]">
                    ₹{p.price.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                  <div className="font-mono text-[11px]" style={{ color: pos ? "var(--up)" : "var(--down)" }}>
                    {fmtPct(p.change_pct)}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </Card>
    </Page>
  )
}
