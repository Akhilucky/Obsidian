"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { FlaskConical, Database, BrainCircuit } from "lucide-react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import { api, withTimeout } from "@/lib/api"
import type { TrialsResult, MLabResult, ProviderStatus } from "@/lib/types"

function verdictColor(v: string) {
  if (v === "ADOPT") return { color: "var(--up)", bg: "rgba(52,211,153,0.12)" }
  if (v === "KEEP") return { color: "var(--text-primary)", bg: "rgba(148,163,184,0.12)" }
  if (v === "EQUIVALENT") return { color: "var(--amber)", bg: "rgba(251,191,36,0.12)" }
  return { color: "var(--text-muted)", bg: "rgba(148,163,184,0.08)" }
}

export default function LabPage() {
  const [trials, setTrials] = useState<TrialsResult | null>(null)
  const [mlab, setMlab] = useState<MLabResult | null>(null)
  const [providers, setProviders] = useState<ProviderStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const [t, m, p] = await Promise.all([
        withTimeout(api.trials(), 90000),
        withTimeout(api.mlLab(), 120000),
        withTimeout(api.providers(), 10000),
      ])
      if (!alive) return
      setTrials(t)
      setMlab(m)
      setProviders(p)
      setLoading(false)
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  return (
    <Page
      title="Quant Lab"
      subtitle="Tool trials & model experiments — benchmarked against the production stack"
      badges={[{ label: "LAB", tone: "live" }, { label: "Walk-Forward" }]}
    >
      {/* Data providers */}
      <Card title="Data Providers" badge={<Database size={14} style={{ color: "var(--text-muted)" }} />} className="mb-4">
        {loading && !providers ? (
          <div className="font-mono text-[12px] text-[var(--text-muted)]">Loading provider status…</div>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {(providers?.providers ?? []).map((p) => (
              <div
                key={p.name}
                className="rounded-xl border p-3"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[13px] font-semibold text-[var(--text-primary)]">
                    {p.name}
                  </span>
                  <span
                    className="rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider"
                    style={
                      p.configured
                        ? { background: "rgba(52,211,153,0.12)", color: "var(--up)" }
                        : { background: "rgba(148,163,184,0.1)", color: "var(--text-muted)" }
                    }
                  >
                    {p.configured ? "active" : "no key"}
                  </span>
                </div>
                {p.key_env && (
                  <div className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">
                    {p.key_env}
                    {p.daily_limit ? ` · ${p.used_today}/${p.daily_limit}/day` : ""}
                  </div>
                )}
                <div className="mt-1 font-mono text-[11px]" style={{ color: "var(--text-muted)" }}>
                  hits {p.hits} · errors {p.errors}
                  {p.last_error && (
                    <span className="block truncate text-[10px]" style={{ color: "var(--down)" }}>
                      {p.last_error}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {!providers && (
              <div className="font-mono text-[12px] text-[var(--text-muted)]">Providers unavailable</div>
            )}
          </div>
        )}
      </Card>

      {/* Tool trials */}
      <Card title="Tool Trials" badge={<FlaskConical size={14} style={{ color: "var(--text-muted)" }} />} className="mb-4">
        <div className="mb-3 font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
          Candidate tools benchmarked against current implementations · verdicts drive auto-integration
        </div>
        {loading && !trials ? (
          <div className="py-8 text-center font-mono text-[12px] text-[var(--text-muted)]">
            Running trials — fetching benchmark data…
          </div>
        ) : (
          <div className="space-y-2">
            {(trials?.results ?? []).map((t) => {
              const vc = verdictColor(t.verdict)
              return (
                <motion.div
                  key={t.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-xl border p-3"
                  style={{ borderColor: "var(--border)" }}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-[13px] font-semibold text-[var(--text-primary)]">
                      {t.name}
                    </span>
                    <span
                      className="rounded-full px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider"
                      style={{ background: vc.bg, color: vc.color }}
                    >
                      {t.verdict}
                    </span>
                    <span className="font-mono text-[11px] text-[var(--text-muted)]">
                      {t.candidate} → {t.incumbent}
                    </span>
                  </div>
                  <div className="mt-1.5 text-[12px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                    {t.notes}
                  </div>
                  {Object.keys(t.latency).length > 0 && (
                    <div className="mt-1.5 font-mono text-[11px]" style={{ color: "var(--text-muted)" }}>
                      {Object.entries(t.latency)
                        .map(([k, v]) => `${k}: ${typeof v === "number" ? (v as number).toFixed(2) : v}s`)
                        .join(" · ")}
                    </div>
                  )}
                </motion.div>
              )
            })}
            {trials?.error && (
              <div className="font-mono text-[12px]" style={{ color: "var(--down)" }}>{trials.error}</div>
            )}
          </div>
        )}
      </Card>

      {/* ML strategy lab */}
      <Card title="ML Strategy Lab" badge={<BrainCircuit size={14} style={{ color: "var(--text-muted)" }} />}>
        <div className="mb-3 flex flex-wrap items-center gap-3 font-mono text-[11px] uppercase tracking-widest text-[var(--text-muted)]">
          <span>{mlab ? `${mlab.samples} samples · ${mlab.folds} folds · ${mlab.horizon_days}d horizon` : "running…"}</span>
          {mlab && <span>Fama-French factors: {mlab.ff_factor_rows}</span>}
          {mlab && <span>took {mlab.took_s}s</span>}
        </div>
        {loading && !mlab ? (
          <div className="py-8 text-center font-mono text-[12px] text-[var(--text-muted)]">
            Training walk-forward models — this takes ~30s…
          </div>
        ) : (
          <>
            <table className="obs-table w-full">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Acc</th>
                  <th>F1</th>
                  <th>Strat Ret</th>
                  <th>Sharpe</th>
                  <th>Max DD</th>
                  <th>Beats Ensemble</th>
                </tr>
              </thead>
              <tbody>
                {mlab &&
                  Object.entries(mlab.models).map(([name, m]) => (
                    <tr key={name}>
                      <td className="font-mono text-[12px] text-[var(--text-primary)]">{name}</td>
                      <td className="font-mono">{m.accuracy.toFixed(3)}</td>
                      <td className="font-mono">{m.f1.toFixed(3)}</td>
                      <td className="font-mono" style={{ color: m.strategy_return >= 0 ? "var(--up)" : "var(--down)" }}>
                        {m.strategy_return.toFixed(1)}%
                      </td>
                      <td className="font-mono">{m.sharpe.toFixed(2)}</td>
                      <td className="font-mono">{m.max_drawdown.toFixed(1)}%</td>
                      <td>
                        {m.outperforms_ensemble === null ? (
                          <span className="font-mono text-[11px] text-[var(--text-muted)]">—</span>
                        ) : m.outperforms_ensemble ? (
                          <span className="font-mono text-[11px] font-bold" style={{ color: "var(--up)" }}>YES</span>
                        ) : (
                          <span className="font-mono text-[11px] text-[var(--text-muted)]">no</span>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {mlab && (
              <div
                className="mt-4 rounded-xl border p-3 font-mono text-[12px]"
                style={{ borderColor: "var(--border)", color: "var(--text-primary)" }}
              >
                <span className="font-bold" style={{ color: "var(--amber)" }}>Winner: {mlab.winner}</span>
                {" — "}
                {mlab.verdict}
              </div>
            )}
          </>
        )}
      </Card>
    </Page>
  )
}
