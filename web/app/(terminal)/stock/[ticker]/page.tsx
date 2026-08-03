"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { motion } from "framer-motion"
import { ArrowLeft, BarChart3, TrendingDown, TrendingUp } from "lucide-react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import PriceChart from "@/components/charts/PriceChart"
import PinButton from "@/components/ui/PinButton"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { api, withTimeout } from "@/lib/api"
import type {
  AnalystsResult,
  ChartPoint,
  DividendsResult,
  EarningsResult,
  FundamentalsResult,
  NewsResult,
  OptionsResult,
  PeersResult,
  Quote,
} from "@/lib/types"
import { fmtNum, fmtPct } from "@/lib/format"

export default function StockPage() {
  const params = useParams<{ ticker: string }>()
  const ticker = (params?.ticker ?? "AAPL").toUpperCase()

  const [quote, setQuote] = useState<Quote | null>(null)
  const [points, setPoints] = useState<ChartPoint[]>([])
  const [analysts, setAnalysts] = useState<AnalystsResult | null>(null)
  const [earnings, setEarnings] = useState<EarningsResult | null>(null)
  const [news, setNews] = useState<NewsResult | null>(null)
  const [fundamentals, setFundamentals] = useState<FundamentalsResult | null>(null)
  const [peers, setPeers] = useState<PeersResult | null>(null)
  const [dividends, setDividends] = useState<DividendsResult | null>(null)
  const [options, setOptions] = useState<OptionsResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [q, c, a, e, n, f, p, dv, o] = await Promise.all([
        withTimeout(api.quote(ticker), 15000),
        withTimeout(api.chart(ticker, "6mo"), 15000),
        withTimeout(api.analysts(ticker), 15000),
        withTimeout(api.earnings(ticker), 15000),
        withTimeout(api.news(ticker, 15), 15000),
        withTimeout(api.fundamentals(ticker), 20000),
        withTimeout(api.peers(ticker), 20000),
        withTimeout(api.dividends(ticker), 15000),
        withTimeout(api.options(ticker), 20000),
      ])
      if (!alive) return
      setQuote(q)
      setPoints(c?.points ?? [])
      setAnalysts(a)
      setEarnings(e)
      setNews(n)
      setFundamentals(f)
      setPeers(p)
      setDividends(dv)
      setOptions(o)
      setLoading(false)
    }
    load()
    const id = setInterval(load, 120000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [ticker])

  const up = (quote?.change_pct ?? 0) >= 0
  const latest = analysts?.periods?.[0]

  return (
    <Page
      title={ticker}
      subtitle={quote?.name ?? "Loading…"}
      badges={[
        { label: "Intelligence", tone: "live" },
        ...(analysts?.consensus_label ? [{ label: analysts.consensus_label }] : []),
      ]}
    >
      {/* Header quote */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-5 flex flex-wrap items-end justify-between gap-4"
      >
        <div className="flex items-end gap-4">
          <Link
            href="/overview"
            className="mb-1 rounded-lg p-2 text-[var(--text-muted)] transition-colors hover:bg-[var(--hover)] hover:text-[var(--text-primary)]"
            aria-label="Back"
          >
            <ArrowLeft size={17} />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[30px] font-bold leading-none text-[var(--text-primary)]">
                {quote ? quote.price.toFixed(2) : loading ? "—" : "—"}
              </span>
              <span
                className="font-mono text-[14px]"
                style={{ color: up ? "var(--up)" : "var(--down)" }}
              >
                {quote ? fmtPct(quote.change_pct) : ""}
              </span>
            </div>
            <div className="mt-1.5 font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
              {quote?.sector ?? "—"} · {quote?.open ? `Open ${quote.open.toFixed(2)}` : ""}
            </div>
          </div>
        </div>
        <PinButton symbol={ticker} size={17} />
      </motion.div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <Card
          title={`${ticker} — 6 Month`}
          badge={
            <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>
              Price
            </span>
          }
          className="hud-corners xl:col-span-2"
        >
          {loading ? (
            <div className="skeleton h-[340px] w-full" />
          ) : (
            <PriceChart points={points} height={340} />
          )}
        </Card>

        <Card title="Key Statistics" className="xl:col-span-1">
          <div className="space-y-2.5">
            {[
              ["Volume", quote ? fmtNum(quote.volume, "") : "—"],
              ["Market Cap", quote ? fmtNum(quote.market_cap) : "—"],
              ["P/E Ratio", quote?.pe_ratio ? quote.pe_ratio.toFixed(2) : "—"],
              ["Day Range", quote ? `${quote.low.toFixed(2)} – ${quote.high.toFixed(2)}` : "—"],
              ["Prev Close", quote ? quote.prev_close.toFixed(2) : "—"],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between border-b pb-2.5 text-[12px]" style={{ borderColor: "rgba(255,255,255,0.035)" }}>
                <span className="text-[var(--text-muted)]">{k}</span>
                <span className="font-mono text-[var(--text-primary)]">{v}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 xl:grid-cols-3">
        <Card
          title="Analyst Ratings"
          badge={
            latest ? (
              <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider" style={{ borderColor: "var(--border-strong)", color: "var(--accent)" }}>
                {latest.period}
              </span>
            ) : undefined
          }
          className="xl:col-span-1"
        >
          {analysts?.total ? (
            <>
              <div className="mb-4 flex items-end justify-between">
                <div>
                  <div className="font-mono text-[26px] font-bold text-[var(--text-primary)]">
                    {analysts.consensus_label}
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-[var(--text-muted)]">
                    {analysts.total} analysts · score {analysts.consensus?.toFixed(2)}
                  </div>
                </div>
              </div>
              <RatingBars latest={latest} total={analysts.total} />
              <div className="mt-4 grid grid-cols-5 gap-1 text-center">
                {["SB", "B", "H", "S", "SS"].map((k, i) => (
                  <div key={k}>
                    <div className="font-mono text-[10px] text-[var(--text-muted)]">{k}</div>
                    <div className="font-mono text-[13px] text-[var(--text-primary)]">
                      {latest ? Object.values(latest).slice(1)[i] : 0}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">
              {analysts?.error ? "Ratings unavailable" : loading ? "Loading…" : "No analyst coverage"}
            </div>
          )}
        </Card>

        <Card
          title="Recent Earnings"
          badge={
            earnings?.next_earnings ? (
              <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider" style={{ borderColor: "var(--border-strong)", color: "var(--amber)" }}>
                Next {earnings.next_earnings}
              </span>
            ) : undefined
          }
          className="xl:col-span-2"
        >
          {earnings?.quarters?.length ? (
            <div className="overflow-x-auto">
              <table className="obs-table w-full">
                <thead>
                  <tr>
                    <th>Quarter End</th><th>EPS Estimate</th><th>Reported EPS</th><th>Surprise</th>
                  </tr>
                </thead>
                <tbody>
                  {earnings.quarters.slice(0, 8).map((q) => {
                    const surp = q.surprise_pct
                    return (
                      <tr key={q.date}>
                        <td className="font-mono text-[var(--text-primary)]">{q.date}</td>
                        <td className="font-mono">{q.eps_estimate?.toFixed(2) ?? "—"}</td>
                        <td className="font-mono">{q.eps_actual?.toFixed(2) ?? "—"}</td>
                        <td
                          className="font-mono"
                          style={{ color: surp == null ? "var(--text-muted)" : surp >= 0 ? "var(--up)" : "var(--down)" }}
                        >
                          {surp == null ? "—" : `${surp >= 0 ? "+" : ""}${surp.toFixed(1)}%`}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {typeof earnings.beat_rate === "number" && (
                <div className="mt-3 flex items-center gap-2 font-mono text-[11px] text-[var(--text-muted)]">
                  Beat rate (last 12 quarters):{" "}
                  <span className="text-[var(--up)]">{(earnings.beat_rate * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>
          ) : (
            <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">
              {earnings?.error ? "Earnings unavailable" : loading ? "Loading…" : "No earnings data"}
            </div>
          )}
        </Card>
      </div>

      <Card title="News & Sentiment" className="mt-5">
        {news?.items?.length ? (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <SentimentChip label={news.net_label} score={news.net_sentiment} big />
            <span className="font-mono text-[11px] text-[var(--text-muted)]">
              Net sentiment across {news.items.length} headlines
            </span>
          </div>
        ) : null}
        <Tabs defaultValue="latest" className="mt-2">
          <TabsList className="mb-4 bg-[var(--bg-elevated)]">
            <TabsTrigger value="latest">Latest</TabsTrigger>
            <TabsTrigger value="bullish">Bullish</TabsTrigger>
            <TabsTrigger value="bearish">Bearish</TabsTrigger>
          </TabsList>
          <TabsContent value="latest">
            <NewsList items={news?.items ?? []} />
          </TabsContent>
          <TabsContent value="bullish">
            <NewsList items={(news?.items ?? []).filter((i) => i.label === "bullish")} />
          </TabsContent>
          <TabsContent value="bearish">
            <NewsList items={(news?.items ?? []).filter((i) => i.label === "bearish")} />
          </TabsContent>
        </Tabs>
      </Card>

      {/* Bloomberg-style intelligence tabs: FA / RV / DVD / OMON */}
      <Card title="Bloomberg Intelligence" className="mt-5">
        <Tabs defaultValue="fundamentals">
          <TabsList className="mb-4 bg-[var(--bg-elevated)]">
            <TabsTrigger value="fundamentals">FA · Fundamentals</TabsTrigger>
            <TabsTrigger value="peers">RV · Peers</TabsTrigger>
            <TabsTrigger value="dividends">DVD · Dividends</TabsTrigger>
            <TabsTrigger value="options">OMON · Options</TabsTrigger>
          </TabsList>
          <TabsContent value="fundamentals">
            <FundamentalsView data={fundamentals} loading={loading} />
          </TabsContent>
          <TabsContent value="peers">
            <PeersView data={peers} loading={loading} />
          </TabsContent>
          <TabsContent value="dividends">
            <DividendsView data={dividends} loading={loading} />
          </TabsContent>
          <TabsContent value="options">
            <OptionsView data={options} loading={loading} />
          </TabsContent>
        </Tabs>
      </Card>
    </Page>
  )
}

function RatingBars({ latest, total }: { latest?: AnalystsResult["periods"][0]; total: number }) {
  const rows = [
    ["Strong Buy", "strongBuy", "var(--up)"],
    ["Buy", "buy", "#38bdf8"],
    ["Hold", "hold", "var(--amber)"],
    ["Sell", "sell", "#f97316"],
    ["Strong Sell", "strongSell", "var(--down)"],
  ] as const
  return (
    <div className="space-y-2">
      {rows.map(([label, key, color]) => {
        const val = latest?.[key] ?? 0
        return (
          <div key={key} className="flex items-center gap-2 text-[11px]">
            <span className="w-20 text-[var(--text-muted)]">{label}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${total ? (val / total) * 100 : 0}%`,
                  background: color,
                  boxShadow: `0 0 8px ${color}55`,
                  transition: "width 600ms cubic-bezier(0.32,0.72,0,1)",
                }}
              />
            </div>
            <span className="w-6 text-right font-mono text-[var(--text-primary)]">{val}</span>
          </div>
        )
      })}
    </div>
  )
}

function SentimentChip({ label, score, big = false }: { label: string; score: number; big?: boolean }) {
  const tone =
    label === "bullish" ? "var(--up)" : label === "bearish" ? "var(--down)" : "var(--text-muted)"
  const Icon = label === "bullish" ? TrendingUp : label === "bearish" ? TrendingDown : BarChart3
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-mono uppercase tracking-wider ${
        big ? "px-3 py-1 text-[11px]" : "px-2 py-0.5 text-[9px]"
      }`}
      style={{ color: tone, borderColor: `${tone}44`, background: `${tone}11` }}
    >
      <Icon size={big ? 13 : 10} />
      {label} {big ? `· ${score >= 0 ? "+" : ""}${score.toFixed(2)}` : ""}
    </span>
  )
}

function NewsList({ items }: { items: NewsResult["items"] }) {
  if (!items.length)
    return <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">No headlines</div>
  return (
    <div className="space-y-1">
      {items.map((n, i) => (
        <motion.a
          key={`${n.title}-${i}`}
          href={n.link || undefined}
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.03, duration: 0.3 }}
          className="group flex items-start justify-between gap-4 rounded-xl border border-transparent px-3 py-2.5 transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--hover)]"
        >
          <div className="min-w-0">
            <div className="truncate text-[13px] text-[var(--text-secondary)] transition-colors group-hover:text-[var(--text-primary)]">
              {n.title}
            </div>
            <div className="mt-1 flex items-center gap-2 font-mono text-[10px] text-[var(--text-muted)]">
              <span>{n.publisher}</span>
              {n.time > 0 && <span>· {timeAgo(n.time)}</span>}
            </div>
          </div>
          <SentimentChip label={n.label} score={n.sentiment} />
        </motion.a>
      ))}
    </div>
  )
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000) - ts
  if (diff < 3600) return `${Math.max(1, Math.floor(diff / 60))}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function FundamentalsView({ data, loading }: { data: FundamentalsResult | null; loading: boolean }) {
  if (loading && !data) return <div className="skeleton h-56 w-full" />
  if (!data?.income?.length && !data?.snapshot) {
    return <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">Fundamentals unavailable</div>
  }
  const snap = data?.snapshot
  return (
    <div className="space-y-5">
      {snap && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            ["Beta", snap.beta ? snap.beta.toFixed(2) : "—"],
            ["52W Range", snap.fifty_two_week_low && snap.fifty_two_week_high ? `${snap.fifty_two_week_low.toFixed(0)} – ${snap.fifty_two_week_high.toFixed(0)}` : "—"],
            ["Target Price", snap.target_mean_price ? `$${snap.target_mean_price.toFixed(2)}` : "—"],
            ["Fwd P/E", snap.forward_pe ? snap.forward_pe.toFixed(1) : "—"],
            ["Div Yield", snap.dividend_yield ? `${snap.dividend_yield.toFixed(2)}%` : "—"],
            ["Profit Margin", snap.profit_margin ? `${snap.profit_margin.toFixed(1)}%` : "—"],
            ["ROE", snap.return_on_equity ? `${snap.return_on_equity.toFixed(1)}%` : "—"],
            ["Shares Out", snap.shares_outstanding ? fmtNum(snap.shares_outstanding) : "—"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-xl border p-3" style={{ borderColor: "var(--border)" }}>
              <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{k}</div>
              <div className="mt-1 font-mono text-[15px] text-[var(--text-primary)]">{v}</div>
            </div>
          ))}
        </div>
      )}
      <FundTable title="Income Statement" groups={data?.income ?? []} />
      <FundTable title="Balance Sheet" groups={data?.balance ?? []} />
      <FundTable title="Cash Flow" groups={data?.cashflow ?? []} />
    </div>
  )
}

function FundTable({ title, groups }: { title: string; groups: FundamentalsResult["income"] }) {
  if (!groups.length) return null
  const years = groups[0].values.map((v) => v.date.slice(0, 10))
  return (
    <div>
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">{title}</div>
      <table className="obs-table w-full">
        <thead>
          <tr>
            <th>Metric</th>
            {years.map((y) => <th key={y}>{y}</th>)}
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g.metric}>
              <td className="text-[var(--text-secondary)]">{g.metric}</td>
              {g.values.map((v, i) => (
                <td key={i} className="font-mono text-[var(--text-primary)]">
                  {Math.abs(v.value) >= 1e9 ? fmtNum(v.value) : v.value.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PeersView({ data, loading }: { data: PeersResult | null; loading: boolean }) {
  if (loading && !data) return <div className="skeleton h-48 w-full" />
  if (!data?.peers?.length) return <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">No comparable peers found</div>
  return (
    <div>
      <div className="mb-3 font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
        Sector: {data.sector} · Relative valuation
      </div>
      <table className="obs-table w-full">
        <thead>
          <tr>
            <th>Symbol</th><th>Price</th><th>Change</th><th>P/E</th><th>Mkt Cap</th>
          </tr>
        </thead>
        <tbody>
          {data.peers.map((p) => (
            <tr key={p.symbol} className="group">
              <td className="font-mono font-semibold text-[var(--text-primary)]">
                <Link href={`/stock/${p.symbol}`} className="transition-colors hover:text-[var(--accent)]">
                  {p.symbol}
                </Link>
              </td>
              <td className="font-mono">{p.price.toFixed(2)}</td>
              <td className="font-mono" style={{ color: p.change_pct >= 0 ? "var(--up)" : "var(--down)" }}>
                {fmtPct(p.change_pct)}
              </td>
              <td className="font-mono">{p.pe_ratio ? p.pe_ratio.toFixed(1) : "—"}</td>
              <td className="font-mono">{p.market_cap ? fmtNum(p.market_cap) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DividendsView({ data, loading }: { data: DividendsResult | null; loading: boolean }) {
  if (loading && !data) return <div className="skeleton h-48 w-full" />
  if (!data?.history?.length) return <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">No dividend history</div>
  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
      <div className="space-y-3">
        <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)" }}>
          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Annual Yield</div>
          <div className="mt-1 font-mono text-[22px] font-bold text-[var(--text-primary)]">
            {data.yield ? `${data.yield.toFixed(2)}%` : "—"}
          </div>
        </div>
        <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)" }}>
          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Payout Ratio</div>
          <div className="mt-1 font-mono text-[22px] font-bold text-[var(--text-primary)]">
            {data.payout_ratio ? `${data.payout_ratio.toFixed(1)}%` : "—"}
          </div>
        </div>
      </div>
      <table className="obs-table w-full md:col-span-2">
        <thead>
          <tr><th>Ex-Date</th><th>Dividend / Share</th></tr>
        </thead>
        <tbody>
          {data.history.slice().reverse().map((h) => (
            <tr key={h.date}>
              <td className="font-mono text-[var(--text-primary)]">{h.date}</td>
              <td className="font-mono">${h.amount.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function OptionsView({ data, loading }: { data: OptionsResult | null; loading: boolean }) {
  if (loading && !data) return <div className="skeleton h-48 w-full" />
  const calls = data?.chain?.calls ?? []
  const puts = data?.chain?.puts ?? []
  if (!calls.length && !puts.length) {
    return <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">No options chain available</div>
  }
  return (
    <div>
      <div className="mb-3 font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
        Expiry {data?.expiry ?? "—"} {data?.underlying ? `· Spot $${data.underlying.toFixed(2)}` : ""}
      </div>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {([["CALLS", calls], ["PUTS", puts]] as [string, OptionsResult["chain"]["calls"]][]).map(([label, rows]) => (
          <div key={label}>
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider" style={{ color: label === "CALLS" ? "var(--up)" : "var(--down)" }}>
              {label}
            </div>
            <table className="obs-table w-full">
              <thead>
                <tr><th>Strike</th><th>Last</th><th>Bid/Ask</th><th>IV</th><th>OI</th></tr>
              </thead>
              <tbody>
                {rows.map((o, i) => (
                  <tr key={`${label}-${i}`}>
                    <td className="font-mono text-[var(--text-primary)]">{o.strike.toFixed(0)}</td>
                    <td className="font-mono">{o.last_price.toFixed(2)}</td>
                    <td className="font-mono text-[11px]">{o.bid.toFixed(2)} / {o.ask.toFixed(2)}</td>
                    <td className="font-mono">{(o.implied_vol * 100).toFixed(0)}%</td>
                    <td className="font-mono">{o.open_interest ? Math.round(o.open_interest).toLocaleString() : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
