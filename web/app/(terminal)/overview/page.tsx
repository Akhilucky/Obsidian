"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import Page from "@/components/shell/Page"
import TickerStrip from "@/components/ui/TickerStrip"
import MetricCard from "@/components/ui/MetricCard"
import Card from "@/components/ui/Card"
import PriceChart from "@/components/charts/PriceChart"
import StatusDot from "@/components/ui/StatusDot"
import PinButton from "@/components/ui/PinButton"
import FocusSelector from "@/components/ui/FocusSelector"
import { useWatchlist } from "@/lib/watchlist-store"
import { api, withTimeout } from "@/lib/api"
import type { ChartPoint, IndexQuote, MoversResult, Quote } from "@/lib/types"
import { fmtNum, fmtPct } from "@/lib/format"

export default function OverviewPage() {
  const [indices, setIndices] = useState<Record<string, IndexQuote> | null>(null)
  const [focus, setFocus] = useState<string | null>(null)
  const [focusQuote, setFocusQuote] = useState<Quote | null>(null)
  const [sp500, setSp500] = useState<ChartPoint[]>([])
  const [movers, setMovers] = useState<MoversResult | null>(null)
  const [loading, setLoading] = useState(true)
  const watchlist = useWatchlist((s) => s.symbols)
  const loadWatchlist = useWatchlist((s) => s.load)

  useEffect(() => {
    loadWatchlist()
    api.focus().then((f) => {
      if (f && f.ticker) setFocus(f.ticker)
    })
    let alive = true
    const load = async () => {
      const [idx, chart, mov] = await Promise.all([
        withTimeout(api.indices("us"), 12000),
        withTimeout(api.chart("^GSPC", "6mo"), 12000),
        withTimeout(api.movers("us"), 20000),
      ])
      if (!alive) return
      setIndices(idx)
      setSp500(chart?.points ?? [])
      setMovers(mov)
      setLoading(false)
    }
    load()
    const id = setInterval(load, 60000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [loadWatchlist])

  useEffect(() => {
    if (!focus) return
    let alive = true
    const loadQuote = async () => {
      const [q, c] = await Promise.all([
        withTimeout(api.quote(focus), 15000),
        withTimeout(api.chart(focus, "6mo"), 15000),
      ])
      if (!alive) return
      setFocusQuote(q)
      setSp500(c?.points ?? [])
    }
    loadQuote()
    return () => {
      alive = false
    }
  }, [focus])

  const onFocusChange = (t: string) => {
    setFocus(t)
    api.setFocus(t)
  }

  const last = sp500[sp500.length - 1]
  const first = sp500[0]
  const pct = last && first ? ((last.close - first.close) / first.close) * 100 : 0
  const up = pct >= 0
  const fqUp = (focusQuote?.change_pct ?? 0) >= 0

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
          title={focus ? `${focus} — 6 Month` : "S&P 500 — 6 Month"}
          badge={
            focus ? (
              <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>
                Focus
              </span>
            ) : (
              <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>
                Index
              </span>
            )
          }
          actions={<FocusSelector ticker={focus ?? "^GSPC"} onChange={onFocusChange} />}
          className="hud-corners xl:col-span-2 obs-card-hover"
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
                  {focus ? "Focus 6M Performance" : "6M Performance"}
                </span>
                {focus && focusQuote && (
                  <>
                    <span className="text-[var(--border-strong)]">|</span>
                    <span className="font-mono text-[11px] text-[var(--text-muted)]">{focusQuote.name}</span>
                    <span className="font-mono text-[13px]" style={{ color: fqUp ? "var(--up)" : "var(--down)" }}>
                      {fmtPct(focusQuote.change_pct)}
                    </span>
                    <Link
                      href={`/stock/${focus}`}
                      className="ml-1 rounded-lg border px-2.5 py-0.5 text-[10px] uppercase tracking-wider text-[var(--accent)] transition-all hover:border-[var(--accent)]"
                      style={{ borderColor: "var(--border-strong)" }}
                    >
                      Full Profile →
                    </Link>
                  </>
                )}
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

          <Card title="Market Movers">
            <MoversList movers={movers} loading={loading} />
          </Card>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard label="S&P 500" value={indices?.["S&P 500"]?.price ?? 0} prefix="" delta={indices?.["S&P 500"]?.change_pct} delay={0} />
        <MetricCard label="NASDAQ" value={indices?.NASDAQ?.price ?? 0} prefix="" delta={indices?.NASDAQ?.change_pct} delay={0.05} />
        <MetricCard label="DOW 30" value={indices?.["DOW 30"]?.price ?? 0} prefix="" delta={indices?.["DOW 30"]?.change_pct} delay={0.1} />
        <MetricCard label="Russell 2K" value={indices?.["Russell 2K"]?.price ?? 0} prefix="" delta={indices?.["Russell 2K"]?.change_pct} delay={0.15} />
      </div>

      <Card
        title="Watchlist Positions"
        badge={<span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>Pinned</span>}
        className="mt-5"
      >
        {watchlist.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center">
            <div className="font-mono text-[13px] text-[var(--text-muted)]">No pinned stocks yet</div>
            <div className="text-[12px] text-[var(--text-muted)]">
              Pin tickers from any page to build your watchlist
            </div>
          </div>
        ) : (
          <table className="obs-table w-full">
            <thead>
              <tr>
                <th>Symbol</th><th>Name</th><th>Price</th><th>Change</th><th>Volume</th><th>Mkt Cap</th><th></th>
              </tr>
            </thead>
            <tbody>
              {watchlist.map((t) => (
                <WatchRow key={t} ticker={t} />
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </Page>
  )
}

function MoversList({ movers, loading }: { movers: MoversResult | null; loading: boolean }) {
  if (loading && !movers) return <div className="skeleton h-40 w-full" />
  const rows = movers?.gainers.slice(0, 5) ?? []
  return (
    <div className="space-y-2">
      {rows.map((m) => (
        <div key={m.symbol} className="flex items-center justify-between text-[12px]">
          <Link href={`/stock/${m.symbol}`} className="flex items-center gap-2">
            <span className="font-mono font-semibold text-[var(--text-primary)] transition-colors hover:text-[var(--accent)]">
              {m.symbol}
            </span>
            <span className="truncate text-[var(--text-muted)]">{m.name}</span>
          </Link>
          <span className="font-mono" style={{ color: m.change_pct >= 0 ? "var(--up)" : "var(--down)" }}>
            {fmtPct(m.change_pct)}
          </span>
        </div>
      ))}
    </div>
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
    <tr className="group">
      <td className="font-mono font-semibold text-[var(--text-primary)]">
        <Link href={`/stock/${ticker}`} className="transition-colors hover:text-[var(--accent)]">
          {ticker}
        </Link>
      </td>
      <td>{quote?.name ?? "—"}</td>
      <td className="font-mono">{quote ? quote.price.toFixed(2) : "—"}</td>
      <td className="font-mono" style={{ color: up ? "var(--up)" : "var(--down)" }}>
        {quote ? fmtPct(quote.change_pct) : "—"}
      </td>
      <td className="font-mono">{quote ? fmtNum(quote.volume, "") : "—"}</td>
      <td className="font-mono">{quote ? fmtNum(quote.market_cap) : "—"}</td>
      <td className="text-right opacity-0 transition-opacity group-hover:opacity-100">
        <PinButton symbol={ticker} />
      </td>
    </tr>
  )
}
