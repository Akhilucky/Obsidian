"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, withTimeout } from "@/lib/api"
import type { MarketsResult, MoversResult } from "@/lib/types"
import { fmtPct } from "@/lib/format"

export default function MarketsPage() {
  const [markets, setMarkets] = useState<MarketsResult | null>(null)
  const [movers, setMovers] = useState<MoversResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [m, mv] = await Promise.all([
        withTimeout(api.markets(), 30000),
        withTimeout(api.movers("us"), 30000),
      ])
      if (!alive) return
      setMarkets(m)
      setMovers(mv)
      setLoading(false)
    }
    load()
    const id = setInterval(load, 120000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  const groupNames = Object.keys(markets?.groups ?? {})

  return (
    <Page
      title="World Markets"
      subtitle="Global indices, FX, commodities & market movers"
      badges={[{ label: "WEI", tone: "live" }, { label: "28 Instruments" }]}
    >
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <Card title="Global Snapshot" className="hud-corners xl:col-span-2">
          {loading && !markets ? (
            <div className="skeleton h-[360px] w-full" />
          ) : markets ? (
            <Tabs defaultValue={groupNames[0] ?? "United States"}>
              <TabsList className="mb-4 flex-wrap bg-[var(--bg-elevated)]" style={{ height: "auto" }}>
                {groupNames.map((g) => (
                  <TabsTrigger key={g} value={g} className="text-[11px]">
                    {g}
                  </TabsTrigger>
                ))}
              </TabsList>
              {groupNames.map((g) => (
                <TabsContent key={g} value={g}>
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                    {markets.groups[g].map((name, i) => {
                      const q = markets.quotes[name]
                      if (!q) return null
                      const pos = q.change_pct >= 0
                      return (
                        <motion.div
                          key={name}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.03, duration: 0.3 }}
                          className="obs-card obs-card-hover p-3.5"
                        >
                          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                            {name}
                          </div>
                          <div className="mt-1.5 font-mono text-[17px] font-semibold text-[var(--text-primary)]">
                            {q.price ? q.price.toFixed(2) : "—"}
                          </div>
                          <div className="font-mono text-[11px]" style={{ color: pos ? "var(--up)" : "var(--down)" }}>
                            {fmtPct(q.change_pct)}
                          </div>
                        </motion.div>
                      )
                    })}
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          ) : null}
        </Card>

        <div className="flex flex-col gap-5">
          <Card title="Top Gainers">
            <MoverTable rows={movers?.gainers ?? []} loading={loading && !movers} />
          </Card>
          <Card title="Top Losers">
            <MoverTable rows={movers?.losers ?? []} loading={loading && !movers} />
          </Card>
          <Card title="Most Active">
            <MoverTable rows={movers?.most_active ?? []} loading={loading && !movers} volume />
          </Card>
        </div>
      </div>
    </Page>
  )
}

function MoverTable({ rows, loading, volume = false }: { rows: { symbol: string; name: string; price: number; change_pct: number; volume: number }[]; loading: boolean; volume?: boolean }) {
  if (loading) return <div className="skeleton h-40 w-full" />
  return (
    <div className="space-y-2">
      {rows.slice(0, 6).map((m) => (
        <Link key={m.symbol} href={`/stock/${m.symbol}`}>
          <div className="group flex items-center justify-between rounded-lg px-2 py-1.5 text-[12px] transition-colors hover:bg-[var(--hover)]">
            <div>
              <span className="font-mono font-semibold text-[var(--text-primary)] transition-colors group-hover:text-[var(--accent)]">
                {m.symbol}
              </span>
              <span className="ml-2 text-[10px] text-[var(--text-muted)]">{m.name}</span>
            </div>
            <div className="flex items-center gap-3">
              {volume && (
                <span className="font-mono text-[10px] text-[var(--text-muted)]">
                  {(m.volume / 1e6).toFixed(0)}M
                </span>
              )}
              <span className="font-mono" style={{ color: m.change_pct >= 0 ? "var(--up)" : "var(--down)" }}>
                {fmtPct(m.change_pct)}
              </span>
            </div>
          </div>
        </Link>
      ))}
    </div>
  )
}
