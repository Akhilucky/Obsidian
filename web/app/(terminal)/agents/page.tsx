"use client"

import { useEffect, useState } from "react"
import Page from "@/components/shell/Page"
import Card from "@/components/ui/Card"
import StatusDot from "@/components/ui/StatusDot"
import { api, withTimeout } from "@/lib/api"
import type { AgentStatus } from "@/lib/types"

const AGENT_DESC: Record<string, string> = {
  DataIngestionAgent: "Market data ingestion from Yahoo Finance & OpenBB",
  DataQualityAgent: "Data validation, gap detection & anomaly checks",
  FeatureEngineeringAgent: "Technical indicator & feature construction",
  RegimeDetectionAgent: "Market regime classification (trend / range / vol)",
  ModelingAgent: "ML model training & prediction",
  DecisionAgent: "Signal aggregation & trade decision logic",
  RiskAgent: "Portfolio risk & position sizing checks",
  ScenarioAgent: "Scenario & stress-test simulation",
  MonitoringAgent: "Live system health & performance monitoring",
  LifecycleAgent: "Trade lifecycle & journal management",
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentStatus[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      const res = await withTimeout(api.agents(), 15000)
      if (!alive) return
      if (res && res.agents.length > 0) {
        setAgents(res.agents)
      } else {
        setAgents([])
        setError(res?.error ?? null)
      }
    }
    load()
    return () => {
      alive = false
    }
  }, [])

  const active = agents?.filter((a) => a.status === "healthy" || a.status === "ok").length ?? 0

  return (
    <Page
      title="Agents"
      subtitle="Multi-agent AI pipeline orchestration"
      badges={[{ label: "Live", tone: "live" }, { label: "Obsidian Orchestrator" }]}
    >
      {/* Pipeline overview */}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Agents" value={agents?.length ?? 0} suffix="" color="var(--text-primary)" />
        <StatCard label="Healthy" value={active} suffix="" color="var(--up)" />
        <StatCard label="Pipeline Stage" value={agents?.length ?? 0} suffix="/11" color="var(--accent)" />
        <StatCard label="Mode" value={0} suffix="" color="var(--text-muted)" custom="Sequential" />
      </div>

      {/* Pipeline flow */}
      <Card title="Pipeline Architecture" className="mb-5">
        <div className="flex flex-wrap items-center gap-1.5 py-2">
          {AGENTS_PIPELINE.map((name, i) => (
            <div key={name} className="flex items-center">
              <div
                className="rounded-lg border px-2.5 py-1.5 text-[10px] font-medium"
                style={{
                  borderColor: "rgba(56,189,248,0.3)",
                  background: "rgba(56,189,248,0.07)",
                  color: "var(--accent)",
                }}
              >
                {name.replace("Agent", "").replace(/([a-z])([A-Z])/g, "$1 $2")}
              </div>
              {i < AGENTS_PIPELINE.length - 1 && (
                <span className="mx-1 text-[11px] text-[var(--text-muted)]">→</span>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Agent grid */}
      {agents === null && !error ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="obs-card p-5">
              <div className="skeleton h-4 w-28" />
              <div className="skeleton mt-2 h-3 w-full" />
              <div className="skeleton mt-3 h-6 w-16" />
            </div>
          ))}
        </div>
      ) : agents && agents.length > 0 ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {agents.map((a) => {
            const healthy = a.status === "healthy" || a.status === "ok"
            const running = a.status === "running"
            return (
              <div key={a.name} className="obs-card obs-card-hover p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-mono text-[13px] font-semibold text-[var(--text-primary)]">
                      {a.name}
                    </div>
                    <div className="mt-1 text-[11px] leading-relaxed text-[var(--text-muted)]">
                      {AGENT_DESC[a.name] ?? "Pipeline agent"}
                    </div>
                  </div>
                  <StatusDot
                    status={healthy ? "up" : running ? "live" : "neutral"}
                    size={7}
                  />
                </div>
                <div className="mt-4 flex items-center justify-between border-t pt-3" style={{ borderColor: "var(--border)" }}>
                  <span
                    className="rounded-full border px-2 py-0.5 text-[9px] font-medium uppercase tracking-wider"
                    style={{
                      borderColor: healthy ? "rgba(52,200,138,0.35)" : running ? "rgba(56,189,248,0.35)" : "rgba(255,255,255,0.15)",
                      color: healthy ? "var(--up)" : running ? "var(--accent)" : "var(--text-muted)",
                      background: healthy ? "rgba(52,200,138,0.08)" : running ? "rgba(56,189,248,0.08)" : "transparent",
                    }}
                  >
                    {healthy ? "Healthy" : running ? "Running" : a.status}
                  </span>
                  <span className="font-mono text-[11px] text-[var(--text-muted)]">
                    {a.latency_ms > 0 ? `${a.latency_ms}ms` : "—"}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <Card title="Agent Runtime">
          <div className="py-8 text-center">
            <p className="text-[13px] text-[var(--text-secondary)]">
              Agent pipeline is not currently running.
            </p>
            <p className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">
              Start it with: <span className="text-[var(--accent)]">python run.py --run</span>
            </p>
            {error && (
              <p className="mt-3 text-[11px] text-[var(--down)]">Status: {error}</p>
            )}
          </div>
        </Card>
      )}
    </Page>
  )
}

const AGENTS_PIPELINE = [
  "DataIngestionAgent", "DataQualityAgent", "FeatureEngineeringAgent",
  "RegimeDetectionAgent", "ModelingAgent", "DecisionAgent", "RiskAgent",
  "ScenarioAgent", "MonitoringAgent", "LifecycleAgent",
]

function StatCard({ label, value, suffix, color, custom }: { label: string; value: number; suffix: string; color: string; custom?: string }) {
  return (
    <div className="obs-card obs-card-hover p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-1.5 font-mono text-[24px] font-bold" style={{ color }}>
        {custom ?? `${value}${suffix}`}
      </div>
    </div>
  )
}
