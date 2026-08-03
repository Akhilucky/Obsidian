"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import TickerStrip from "@/components/ui/TickerStrip"
import MetricCard from "@/components/ui/MetricCard"
import Card from "@/components/ui/Card"
import PriceChart from "@/components/charts/PriceChart"
import StatusDot from "@/components/ui/StatusDot"
import { api, withTimeout } from "@/lib/api"
import type { ChartPoint, IndexQuote, Quote } from "@/lib/types"
import { fmtNum, fmtPct } from "@/lib/format"

export default function OverviewPage() {
  const [indices, setIndices] = useState<Record<string, IndexQuote> | null>(null)
  const [sp500, setSp500] = useState<ChartPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [idx, chart] = await Promise.all([
        withTimeout(api.indices("us"), 12000),
        withTimeout(api.chart("^GSPC", "6mo"), 12000),
      ])
      if (!alive) return
      setIndices(idx)
      setSp500(chart?.points ?? [])
      setLoading(false)
    }
    load()
    const id = setInterval(load, 60000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const last = sp500[sp500.length - 1]
  const first = sp500[0]
  const pct = last && first ? ((last.close - first.close) / first.close) * 100 : 0
  const up = pct >= 0

  const vix = indices?.["VIX"]
  const tenY = indices?.["10Y Yield"]

  return (
    <Page
      title="Overview"
      subtitle="Institutional market snapshot"
      badges={[{ label: "Live", tone: "live" }, { label: "Real-Time Data" }]}
    >
      <TickerStrip items={indices ?? {}} loading={!indices} />

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <Card
          title="S&P 500 — 6 Month"
          badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>Index</span>}
          className="xl:col-span-2 obs-card-hover"
        >
          {loading ? (
            <div className="skeleton h-[340px] w-full" />
          ) : (
            <>
              <PriceChart points={sp500} height={340} />
              <div className="mt-2 flex items-center gap-3">
                <span className="font-mono text-[22px] font-semibold text-[var(--text-primary)]">
                  {last ? last.close.toFixed(2) : "—"}
                </span>
                <span className="font-mono text-[13px]" style={{ color: up ? "var(--up)" : "var(--down)" }}>
                  {fmtPct(pct)}
                </span>
                <span className="font-mono text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
                  6M Performance
                </span>
              </div>
            </>
          )}
        </Card>

        <div className="flex flex-col gap-5">
          <Card title="Market Volatility">
            {vix ? (
              <div className="flex items-end justify-between">
                <div>
                  <div className="font-mono text-[30px] font-bold text-[var(--text-primary)]">
                    {vix.price.toFixed(2)}
                  </div>
                  <div className="mt-1 font-mono text-[12px]" style={{ color: vix.change_pct >= 0 ? "var(--down)" : "var(--up)" }}>
                    {fmtPct(vix.change_pct)}
                  </div>
                </div>
                <StatusDot status={vix.change_pct >= 0 ? "down" : "up"} size={8} />
              </div>
            ) : (
              <div className="skeleton h-16 w-full" />
            )}
            <div className="mt-4 border-t pt-4" style={{ borderColor: "var(--border)" }}>
              <div className="flex justify-between text-[12px]">
                <span className="text-[var(--text-muted)]">10Y Yield</span>
                <span className="font-mono text-[var(--text-primary)]">
                  {tenY ? `${tenY.price.toFixed(2)}%` : "—"}
                </span>
              </div>
            </div>
          </Card>

          <Card title="Market Breadth">
            <div className="space-y-3">
              {["Advancers", "Decliners", "Unchanged"].map((label, i) => (
                <div key={label} className="flex items-center justify-between text-[12px]">
                  <span className="text-[var(--text-muted)]">{label}</span>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-24 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                      <MotionBar color={i === 0 ? "var(--up)" : i === 1 ? "var(--down)" : "var(--text-muted)"} width={i === 0 ? 58 : i === 1 ? 34 : 8} />
                    </div>
                    <span className="w-10 text-right font-mono text-[var(--text-primary)]">
                      {i === 0 ? "58%" : i === 1 ? "34%" : "8%"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard label="S&P 500" value={last?.close ?? 0} prefix="" delta={pct} delay={0} />
        <MetricCard label="NASDAQ" value={indices?.NASDAQ?.price ?? 0} prefix="" delta={indices?.NASDAQ?.change_pct} delay={0.05} />
        <MetricCard label="DOW 30" value={indices?.["DOW 30"]?.price ?? 0} prefix="" delta={indices?.["DOW 30"]?.change_pct} delay={0.1} />
        <MetricCard label="Russell 2K" value={indices?.["Russell 2K"]?.price ?? 0} prefix="" delta={indices?.["Russell 2K"]?.change_pct} delay={0.15} />
      </div>

      <Card title="Watchlist Positions" className="mt-5">
        <table className="obs-table w-full">
          <thead>
            <tr>
              <th>Symbol</th><th>Name</th><th>Price</th><th>Change</th><th>Volume</th><th>Mkt Cap</th>
            </tr>
          </thead>
          <tbody>
            {["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"].map((t) => (
              <WatchRow key={t} ticker={t} />
            ))}
          </tbody>
        </table>
      </Card>
    </Page>
  )
}

function WatchRow({ ticker }: { ticker: string }) {
  const [quote, setQuote] = useState<Quote | null>(null)
  useEffect(() => {
    let alive = true
    api.quote(ticker).then((q) => {
      if (alive && q.price > 0) setQuote(q)
    })
    return () => {
      alive = false
    }
  }, [ticker])
  const up = (quote?.change_pct ?? 0) >= 0
  return (
    <tr>
      <td className="font-mono font-semibold text-[var(--text-primary)]">{ticker}</td>
      <td>{quote?.name ?? "—"}</td>
      <td className="font-mono">{quote ? quote.price.toFixed(2) : "—"}</td>
      <td className="font-mono" style={{ color: up ? "var(--up)" : "var(--down)" }}>
        {quote ? fmtPct(quote.change_pct) : "—"}
      </td>
      <td className="font-mono">{quote ? fmtNum(quote.volume, "") : "—"}</td>
      <td className="font-mono">{quote ? fmtNum(quote.market_cap) : "—"}</td>
    </tr>
  )
}

function MotionBar({ width, color }: { width: number; color: string }) {
  return (
    <div
      className="h-full rounded-full"
      style={{ width: `${width}%`, background: color, boxShadow: `0 0 8px ${color}55` }}
    />
  )
}
