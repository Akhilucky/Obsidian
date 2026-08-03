"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import Page from "@/components/shell/Page"
import TickerStrip from "@/components/ui/TickerStrip"
import Card from "@/components/ui/Card"
import MetricCard from "@/components/ui/MetricCard"
import PriceChart from "@/components/charts/PriceChart"
import PinButton from "@/components/ui/PinButton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, withTimeout } from "@/lib/api"
import type { IndexQuote, IndiaResult } from "@/lib/types"
import { fmtPct } from "@/lib/format"

export default function IndiaPage() {
  const [indices, setIndices] = useState<Record<string, IndexQuote> | null>(null)
  const [data, setData] = useState<IndiaResult | null>(null)
  const [ticker, setTicker] = useState("RELIANCE.NS")
  const [loading, setLoading] = useState(true)
  const [sector, setSector] = useState("")

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
      setData(india)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [ticker])

  const quote = data?.quote ?? null
  const points = data?.points ?? []
  const sectors = data?.sectors ?? []
  const up = (quote?.change_pct ?? 0) >= 0
  const activeSector = sector || sectors[0]?.name || ""

  return (
    <Page
      title="Indian Markets"
      subtitle="NSE / BSE real-time data & analysis"
      badges={[{ label: "NSE Live", tone: "live" }]}
    >
      <TickerStrip items={indices ?? {}} loading={!indices} />

      <div className="mb-6 flex flex-wrap items-center gap-2">
        {sectors.length > 0 &&
          sectors.map((s) => (
            <button
              key={s.name}
              onClick={() => setSector(s.name)}
              className="rounded-lg border px-3 py-1.5 text-[12px] transition-all duration-150 active:scale-95"
              style={{
                borderColor: s.name === activeSector ? "var(--accent)" : "var(--border-strong)",
                background: s.name === activeSector ? "rgba(56,189,248,0.10)" : "rgba(255,255,255,0.02)",
                color: s.name === activeSector ? "var(--accent)" : "var(--text-secondary)",
                boxShadow: s.name === activeSector ? "0 0 16px rgba(56,189,248,0.15)" : "none",
              }}
            >
              {s.name}
            </button>
          ))}
      </div>

      <Card
        title={`${ticker} — 1 Year`}
        actions={
          <div className="flex items-center gap-2">
            <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>
              NSE
            </span>
            <PinButton symbol={ticker} />
          </div>
        }
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
              <Link
                href={`/stock/${ticker}`}
                className="text-[11px] text-[var(--accent)] transition-opacity hover:opacity-80"
              >
                {quote?.name} →
              </Link>
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

      <Card title={`Sector Universe — ${activeSector}`} className="mt-5">
        {sectors.length > 0 ? (
          <Tabs defaultValue={activeSector} value={activeSector} onValueChange={setSector} className="mt-1">
            <TabsList className="mb-4 flex-wrap bg-[var(--bg-elevated)]" style={{ height: "auto" }}>
              {sectors.map((s) => (
                <TabsTrigger key={s.name} value={s.name} className="text-[11px]">
                  {s.name}
                </TabsTrigger>
              ))}
            </TabsList>
            {sectors.map((s) => (
              <TabsContent key={s.name} value={s.name}>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                  {s.stocks.map((st, i) => (
                    <motion.div
                      key={st.symbol}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03, duration: 0.3 }}
                      className="relative"
                    >
                      <button
                        onClick={() => selectTicker(st.symbol)}
                        className={`obs-card obs-card-hover w-full p-3.5 text-left ${ticker === st.symbol ? "ring-1" : ""}`}
                        style={
                          ticker === st.symbol
                            ? { borderColor: "var(--accent)", boxShadow: "0 0 20px rgba(56,189,248,0.12)" }
                            : undefined
                        }
                      >
                        <div className="font-mono text-[13px] font-semibold text-[var(--text-primary)]">
                          {st.symbol.replace(".NS", "")}
                        </div>
                        <div className="text-[10px] text-[var(--text-muted)]">{st.name}</div>
                      </button>
                      <div className="absolute right-1 top-1">
                        <PinButton symbol={st.symbol} size={13} />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </TabsContent>
            ))}
          </Tabs>
        ) : (
          <div className="py-6 text-center font-mono text-[12px] text-[var(--text-muted)]">
            Loading sector universe…
          </div>
        )}
      </Card>
    </Page>
  )
}
