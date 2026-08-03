"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { FilterX } from "lucide-react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import PinButton from "@/components/ui/PinButton"
import { api, withTimeout } from "@/lib/api"
import type { ScreenerResult } from "@/lib/types"
import { fmtNum, fmtPct } from "@/lib/format"

const SECTORS = [
  "Any", "Technology", "Consumer Discretionary", "Financials", "Healthcare",
  "Industrials", "Energy", "Communication Services", "Consumer Staples",
]

export default function ScreenerPage() {
  const [market, setMarket] = useState<"us" | "india">("us")
  const [sector, setSector] = useState("Any")
  const [minCap, setMinCap] = useState(0)
  const [maxPe, setMaxPe] = useState(0)
  const [data, setData] = useState<ScreenerResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      setLoading(true)
      const r = await withTimeout(
        api.screener({ market, sector: sector === "Any" ? "" : sector, min_cap: minCap, max_pe: maxPe }),
        60000
      )
      if (!alive) return
      setData(r)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [market, sector, minCap, maxPe])

  return (
    <Page
      title="Equity Screener"
      subtitle="Screen the universe by valuation, size & momentum"
      badges={[{ label: "EQS", tone: "live" }, { label: "Universe-Funded" }]}
    >
      {/* Filters */}
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <div className="flex rounded-xl border p-1" style={{ borderColor: "var(--border)" }}>
          {(["us", "india"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMarket(m)}
              className="rounded-lg px-3 py-1.5 text-[12px] font-medium uppercase tracking-wider transition-all duration-150"
              style={{
                background: market === m ? "rgba(56,189,248,0.12)" : "transparent",
                color: market === m ? "var(--accent)" : "var(--text-muted)",
              }}
            >
              {m === "us" ? "US" : "India"}
            </button>
          ))}
        </div>
        <select
          value={sector}
          onChange={(e) => setSector(e.target.value)}
          className="rounded-xl border bg-transparent px-3 py-2 text-[12px] text-[var(--text-secondary)] focus:outline-none"
          style={{ borderColor: "var(--border)" }}
        >
          {SECTORS.map((s) => (
            <option key={s} value={s} className="bg-[#0d0d12]">
              {s}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]">
          Min Market Cap (B$)
          <input
            type="number"
            min={0}
            value={minCap || ""}
            onChange={(e) => setMinCap(Number(e.target.value) * 1e9)}
            placeholder="0"
            className="w-20 rounded-lg border bg-transparent px-2 py-1.5 font-mono text-[12px] focus:outline-none"
            style={{ borderColor: "var(--border)" }}
          />
        </label>
        <label className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]">
          Max P/E
          <input
            type="number"
            min={0}
            value={maxPe || ""}
            onChange={(e) => setMaxPe(Number(e.target.value))}
            placeholder="∞"
            className="w-16 rounded-lg border bg-transparent px-2 py-1.5 font-mono text-[12px] focus:outline-none"
            style={{ borderColor: "var(--border)" }}
          />
        </label>
        <button
          onClick={() => {
            setMinCap(0)
            setMaxPe(0)
            setSector("Any")
          }}
          className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-[11px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
          style={{ borderColor: "var(--border)" }}
        >
          <FilterX size={12} /> Reset
        </button>
      </div>

      <Card
        title="Screen Results"
        badge={
          <span className="rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider text-[var(--text-muted)]" style={{ borderColor: "var(--border-strong)" }}>
            {data?.total ?? "…"} matches
          </span>
        }
      >
        {loading && !data ? (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-11 w-full" />
            ))}
          </div>
        ) : data?.rows?.length ? (
          <table className="obs-table w-full">
            <thead>
              <tr>
                <th></th>
                <th>Symbol</th><th>Name</th><th>Price</th><th>Momentum</th>
                <th>Mkt Cap</th><th>P/E</th><th>Yield</th><th>Sector</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r, i) => (
                <motion.tr
                  key={r.symbol}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.02 }}
                  className="group"
                >
                  <td className="w-8"><PinButton symbol={r.symbol} size={13} /></td>
                  <td className="font-mono font-semibold text-[var(--text-primary)]">
                    <Link href={`/stock/${r.symbol}`} className="transition-colors hover:text-[var(--accent)]">
                      {r.symbol}
                    </Link>
                  </td>
                  <td>{r.name}</td>
                  <td className="font-mono">{r.price ? r.price.toFixed(2) : "—"}</td>
                  <td className="font-mono" style={{ color: r.change_pct >= 0 ? "var(--up)" : "var(--down)" }}>
                    {fmtPct(r.change_pct)}
                  </td>
                  <td className="font-mono">{r.market_cap ? fmtNum(r.market_cap) : "—"}</td>
                  <td className="font-mono">{r.pe_ratio ? r.pe_ratio.toFixed(1) : "—"}</td>
                  <td className="font-mono">{r.dividend_yield ? `${r.dividend_yield.toFixed(2)}%` : "—"}</td>
                  <td className="text-[11px] text-[var(--text-muted)]">{r.sector || "—"}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="py-10 text-center font-mono text-[12px] text-[var(--text-muted)]">
            No results — adjust filters
          </div>
        )}
      </Card>
    </Page>
  )
}
