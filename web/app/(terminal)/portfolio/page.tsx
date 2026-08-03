"use client"

import { useCallback, useEffect, useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { AlertTriangle, Plus, ShieldCheck, Trash2 } from "lucide-react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import AnimatedNumber from "@/components/ui/AnimatedNumber"
import { api, withTimeout } from "@/lib/api"
import type { PortfolioResult } from "@/lib/types"
import { fmtPct } from "@/lib/format"

export default function PortfolioPage() {
  const [data, setData] = useState<PortfolioResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [symbol, setSymbol] = useState("")
  const [qty, setQty] = useState("")
  const [cost, setCost] = useState("")
  const [adding, setAdding] = useState(false)

  const load = useCallback(async () => {
    const r = await withTimeout(api.portfolio(), 45000)
    setData(r)
    setLoading(false)
  }, [])

  useEffect(() => {
    let alive = true
    const doLoad = async () => {
      const r = await withTimeout(api.portfolio(), 45000)
      if (!alive) return
      setData(r)
      setLoading(false)
    }
    doLoad()
    return () => {
      alive = false
    }
  }, [])

  const add = async () => {
    if (!symbol || !qty) return
    setAdding(true)
    await api.portfolioAdd(symbol.trim().toUpperCase(), Number(qty), Number(cost) || 0)
    setSymbol("")
    setQty("")
    setCost("")
    setAdding(false)
    await load()
  }

  const remove = async (sym: string) => {
    await api.portfolioRemove(sym)
    await load()
  }

  const d = data

  return (
    <Page
      title="Portfolio Risk & Analytics"
      subtitle="Aladdin-style position management, risk & stress testing"
      badges={[{ label: "Paper Trading", tone: "live" }, { label: "VaR 95%" }]}
    >
      {/* Header metrics */}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <BigMetric label="Total Value" value={d?.total ?? 0} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Invested" value={d?.invested ?? 0} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Cash" value={d?.cash ?? 0} prefix="$" color="var(--text-primary)" />
        <BigMetric label="Total P&L" value={Math.abs(d?.total_pnl ?? 0)} prefix={(d?.total_pnl ?? 0) >= 0 ? "+$" : "-$"} color={(d?.total_pnl ?? 0) >= 0 ? "var(--up)" : "var(--down)"} sub={fmtPct(d?.total_pnl_pct ?? 0)} />
      </div>

      {/* Risk row */}
      <div className="mb-6 grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card title="Value at Risk (95%, 1-Day)" className="lg:col-span-1">
          {loading && !d ? (
            <div className="skeleton h-40 w-full" />
          ) : (
            <div className="space-y-4 pt-1">
              <div className="flex items-end justify-between">
                <div>
                  <div className="font-mono text-[28px] font-bold text-[var(--text-primary)]">
                    ${(d?.var_95_historical ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-[var(--text-muted)]">
                    {(d?.var_pct_historical ?? 0).toFixed(2)}% of portfolio
                  </div>
                </div>
                <ShieldCheck size={22} style={{ color: "var(--up)" }} />
              </div>
              <div className="border-t pt-3 text-[11px]" style={{ borderColor: "var(--border)" }}>
                <Row label="Parametric (σ)" value={`$${(d?.var_95_parametric ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
                <Row label="Historical" value={`$${(d?.var_95_historical ?? 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
              </div>
            </div>
          )}
        </Card>

        <Card title="Stress Scenarios" className="lg:col-span-2">
          {loading && !d ? (
            <div className="skeleton h-40 w-full" />
          ) : (
            <div className="space-y-2.5 pt-1">
              {(d?.scenarios ?? []).map((s, i) => (
                <motion.div
                  key={s.name}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center justify-between rounded-lg border px-3 py-2"
                  style={{ borderColor: "var(--border)" }}
                >
                  <span className="text-[12px] text-[var(--text-secondary)]">{s.name}</span>
                  <span className="font-mono text-[12px]" style={{ color: "var(--down)" }}>
                    {s.impact < 0 ? "-" : ""}${Math.abs(s.impact).toLocaleString("en-US", { maximumFractionDigits: 0 })} ({fmtPct(s.pct)})
                  </span>
                </motion.div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Allocation + compliance */}
      <div className="mb-6 grid grid-cols-1 gap-5 lg:grid-cols-3">
        <Card title="Sector Exposure" className="lg:col-span-2">
          {loading && !d ? (
            <div className="skeleton h-56 w-full" />
          ) : (
            <div className="space-y-4 pt-2">
              {(d?.sector_exposure ?? []).map((s) => (
                <div key={s.name}>
                  <div className="mb-1.5 flex justify-between text-[12px]">
                    <span className="text-[var(--text-muted)]">{s.name}</span>
                    <span className="font-mono text-[var(--text-primary)]">{s.pct.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${s.pct}%`,
                        background: "linear-gradient(90deg, #38bdf8, #2f6fed)",
                        boxShadow: "0 0 10px rgba(56,189,248,0.35)",
                        transition: "width 800ms ease",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card
          title="Compliance"
          badge={
            (d?.violations.length ?? 0) > 0 ? (
              <span className="flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider" style={{ borderColor: "rgba(228,87,61,0.35)", color: "var(--down)" }}>
                <AlertTriangle size={9} /> {d?.violations.length} breach{d?.violations.length === 1 ? "" : "es"}
              </span>
            ) : (
              <span className="flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-wider" style={{ borderColor: "rgba(52,200,138,0.35)", color: "var(--up)" }}>
                <ShieldCheck size={9} /> Within Limits
              </span>
            )
          }
        >
          {(d?.violations.length ?? 0) > 0 ? (
            <div className="space-y-2 pt-1">
              {d?.violations.map((v, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg border px-3 py-2 text-[12px]"
                  style={{ borderColor: v.severity === "high" ? "rgba(228,87,61,0.3)" : "rgba(217,164,65,0.3)", background: v.severity === "high" ? "rgba(228,87,61,0.06)" : "rgba(217,164,65,0.06)" }}
                >
                  <span className="text-[var(--text-secondary)]">{v.rule}</span>
                  <span className="font-mono" style={{ color: v.severity === "high" ? "var(--down)" : "var(--amber)" }}>
                    {v.limit}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <ShieldCheck size={26} style={{ color: "var(--up)" }} />
              <div className="font-mono text-[12px] text-[var(--text-muted)]">All concentration limits respected</div>
            </div>
          )}
        </Card>
      </div>

      {/* Holdings editor */}
      <Card
        title="Holdings"
        actions={<span className="font-mono text-[10px] uppercase tracking-widest text-[var(--text-muted)]">{d?.position_count ?? 0} positions</span>}
      >
        <div className="mb-4 flex flex-wrap items-end gap-3 rounded-xl border p-3" style={{ borderColor: "var(--border)" }}>
          <Input label="Symbol" value={symbol} onChange={setSymbol} placeholder="AAPL" w="w-28" />
          <Input label="Qty" value={qty} onChange={setQty} placeholder="100" w="w-20" />
          <Input label="Avg Cost" value={cost} onChange={setCost} placeholder="180.00" w="w-24" />
          <button
            onClick={add}
            disabled={adding || !symbol || !qty}
            className="flex items-center gap-1.5 rounded-lg border px-4 py-2 text-[12px] font-medium transition-all duration-150 disabled:opacity-40"
            style={{ borderColor: "var(--border-strong)", background: "rgba(56,189,248,0.08)", color: "var(--accent)" }}
          >
            <Plus size={13} /> {adding ? "Adding…" : "Add Position"}
          </button>
        </div>

        {loading && !d ? (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="skeleton h-11 w-full" />
            ))}
          </div>
        ) : (
          <table className="obs-table w-full">
            <thead>
              <tr>
                <th>Symbol</th><th>Qty</th><th>Avg Cost</th><th>Last</th><th>Value</th><th>P&L</th><th>Weight</th><th></th>
              </tr>
            </thead>
            <tbody>
              {(d?.holdings ?? []).map((h) => {
                const pnl = h.pnl
                const pos = pnl >= 0
                const weight = d?.total ? (h.value / d.total) * 100 : 0
                return (
                  <tr key={h.symbol} className="group">
                    <td className="font-mono font-semibold text-[var(--text-primary)]">
                      <Link href={`/stock/${h.symbol}`} className="transition-colors hover:text-[var(--accent)]">
                        {h.symbol}
                      </Link>
                    </td>
                    <td className="font-mono">{h.qty}</td>
                    <td className="font-mono">${h.avg_cost.toFixed(2)}</td>
                    <td className="font-mono">${h.last.toFixed(2)}</td>
                    <td className="font-mono text-[var(--text-primary)]">${h.value.toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                    <td className="font-mono" style={{ color: pos ? "var(--up)" : "var(--down)" }}>
                      {pos ? "+" : ""}${pnl.toLocaleString("en-US", { maximumFractionDigits: 0 })} ({fmtPct(h.pnl_pct)})
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                          <div className="h-full rounded-full" style={{ width: `${Math.min(100, weight * 6)}%`, background: "var(--accent)" }} />
                        </div>
                        <span className="font-mono text-[11px] text-[var(--text-muted)]">{weight.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="text-right">
                      <button
                        onClick={() => remove(h.symbol)}
                        aria-label={`Remove ${h.symbol}`}
                        className="rounded-lg p-1.5 text-[var(--text-muted)] opacity-0 transition-all hover:text-[var(--down)] group-hover:opacity-100"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Card>
    </Page>
  )
}

function BigMetric({ label, value, prefix, color, sub }: { label: string; value: number; prefix: string; color: string; sub?: string }) {  return (
    <div className="obs-card obs-card-hover p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-1.5 font-mono text-[26px] font-bold" style={{ color }}>
        <AnimatedNumber
          value={value}
          duration={700}
          format={(v) => `${prefix}${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
        />
      </div>
      {sub && <div className="mt-0.5 font-mono text-[12px]" style={{ color }}>{sub}</div>}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="mb-1.5 flex justify-between">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className="font-mono text-[var(--text-primary)]">{value}</span>
    </div>
  )
}

function Input({ label, value, onChange, placeholder, w }: { label: string; value: string; onChange: (v: string) => void; placeholder: string; w: string }) {
  return (
    <label className={`flex flex-col gap-1 ${w}`}>
      <span className="text-[9px] uppercase tracking-widest text-[var(--text-muted)]">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="rounded-lg border bg-transparent px-2.5 py-1.5 font-mono text-[12px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
        style={{ borderColor: "var(--border)" }}
      />
    </label>
  )
}
