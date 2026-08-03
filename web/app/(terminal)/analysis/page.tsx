"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import PriceChart from "@/components/charts/PriceChart"
import IndicatorChart from "@/components/charts/IndicatorChart"
import MetricCard from "@/components/ui/MetricCard"
import AnimatedNumber from "@/components/ui/AnimatedNumber"
import { api, withTimeout } from "@/lib/api"
import type { ChartPoint, Quote } from "@/lib/types"
import { fmtNum, fmtPct } from "@/lib/format"

const TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "AMD"]
const PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]

export default function AnalysisPage() {
  const [ticker, setTicker] = useState("AAPL")
  const [period, setPeriod] = useState("1y")
  const [quote, setQuote] = useState<Quote | null>(null)
  const [points, setPoints] = useState<ChartPoint[]>([])
  const [loading, setLoading] = useState(true)

  const selectTicker = (t: string) => {
    setTicker(t)
    setLoading(true)
  }

  const selectPeriod = (p: string) => {
    setPeriod(p)
    setLoading(true)
  }

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [q, c] = await Promise.all([
        withTimeout(api.quote(ticker), 12000),
        withTimeout(api.chart(ticker, period), 12000),
      ])
      if (!alive) return
      if (q) setQuote(q)
      if (c) setPoints(c.points)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [ticker, period])

  const last = points[points.length - 1]
  const up = (quote?.change_pct ?? 0) >= 0

  return (
    <Page
      title="Analysis"
      subtitle="Deep technical & fundamental research"
      badges={[{ label: "Live", tone: "live" }, { label: ticker }]}
    >
      {/* Controls */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-1.5">
          {TICKERS.map((t) => (
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
              {t}
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-1.5">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => selectPeriod(p)}
              className="rounded-lg px-3 py-1.5 font-mono text-[11px] transition-all duration-150 active:scale-95"
              style={{
                background: p === period ? "rgba(255,255,255,0.08)" : "transparent",
                color: p === period ? "var(--text-primary)" : "var(--text-muted)",
              }}
            >
              {p.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Quote header */}
      <div className="mb-5 flex flex-wrap items-end gap-6">
        <div>
          <div className="text-[11px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
            {quote?.name ?? "Loading…"}
          </div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="font-mono text-[34px] font-bold text-[var(--text-primary)]">
              <AnimatedNumber
                value={quote?.price ?? 0}
                format={(v) => fmtNum(v, "$")}
                duration={700}
              />
            </span>
            <span className="font-mono text-[15px]" style={{ color: up ? "var(--up)" : "var(--down)" }}>
              {quote ? `${quote.change >= 0 ? "+" : ""}${quote.change.toFixed(2)}` : ""}{" "}
              {quote ? fmtPct(quote.change_pct) : ""}
            </span>
          </div>
        </div>
        <div className="flex gap-6 text-[12px]">
          <KV label="Sector" value={quote?.sector ?? "—"} />
          <KV label="P/E" value={quote?.pe_ratio ? quote.pe_ratio.toFixed(1) : "—"} />
          <KV label="Volume" value={quote ? fmtNum(quote.volume, "") : "—"} />
          <KV label="Mkt Cap" value={quote ? fmtNum(quote.market_cap) : "—"} />
          <KV label="Day Range" value={quote ? `${quote.low.toFixed(2)} – ${quote.high.toFixed(2)}` : "—"} />
        </div>
      </div>

      {/* Main chart */}
      <Card
        title={`${ticker} — ${period.toUpperCase()}`}
        badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>SMA + Bollinger</span>}
        className="obs-card-hover"
      >
        {loading ? (
          <div className="skeleton h-[360px] w-full" />
        ) : (
          <PriceChart points={points} height={360} showSMA showBands showEMA={false} />
        )}
      </Card>

      {/* Indicator panels */}
      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card title="Relative Strength Index" badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>14</span>}>
          {loading ? (
            <div className="skeleton h-[140px] w-full" />
          ) : (
            <>
              <IndicatorChart points={points} kind="rsi" />
              <RSIBadges rsi={last?.rsi} />
            </>
          )}
        </Card>
        <Card title="MACD Momentum" badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>12 · 26 · 9</span>}>
          {loading ? (
            <div className="skeleton h-[140px] w-full" />
          ) : (
            <IndicatorChart points={points} kind="macd" />
          )}
        </Card>
      </div>

      {/* Metrics */}
      <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard label="ATR (14)" value={last?.atr ?? 0} prefix="" delta={undefined} />
        <MetricCard label="SMA 20" value={last?.sma_20 ?? 0} prefix="" />
        <MetricCard label="SMA 50" value={last?.sma_50 ?? 0} prefix="" />
        <MetricCard label="BB Width" value={last?.bb_upper && last?.bb_lower ? ((last.bb_upper - last.bb_lower) / (last.bb_middle ?? 1)) * 100 : 0} prefix="" />
      </div>
    </Page>
  )
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-0.5 font-mono text-[13px] text-[var(--text-primary)]">{value}</div>
    </div>
  )
}

function RSIBadges({ rsi }: { rsi?: number }) {
  if (rsi === undefined || Number.isNaN(rsi)) return null
  const zone =
    rsi >= 70 ? { label: "Overbought", color: "var(--down)" }
    : rsi <= 30 ? { label: "Oversold", color: "var(--up)" }
    : { label: "Neutral", color: "var(--text-muted)" }
  return (
    <div className="mt-3 flex items-center gap-3">
      <span className="font-mono text-[20px] font-semibold text-[var(--text-primary)]">{rsi.toFixed(1)}</span>
      <span
        className="rounded-full border px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider"
        style={{ borderColor: zone.color + "44", color: zone.color, background: zone.color + "0d" }}
      >
        {zone.label}
      </span>
    </div>
  )
}
