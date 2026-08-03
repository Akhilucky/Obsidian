import { motion } from "framer-motion"
import AnimatedNumber from "./AnimatedNumber"

type Props = {
  label: string
  value: number
  prefix?: string
  delta?: number
  deltaPct?: number
  delay?: number
  currency?: string
}

export default function MetricCard({
  label,
  value,
  prefix = "$",
  delta,
  deltaPct,
  delay = 0,
  currency = "$",
}: Props) {
  const positive = (delta ?? 0) >= 0
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut", delay }}
      className="obs-card obs-card-hover p-4"
    >
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
        {label}
      </div>
      <div className="mt-1.5 font-mono text-[22px] font-semibold text-[var(--text-primary)]">
        <AnimatedNumber value={value} format={(v) => `${prefix}${v.toFixed(2)}`} />
      </div>
      {delta !== undefined && (
        <div className="mt-1 flex items-center gap-2">
          <span
            className="font-mono text-[12px]"
            style={{ color: positive ? "var(--up)" : "var(--down)" }}
          >
            {positive ? "+" : ""}
            {delta.toFixed(2)}
            {deltaPct !== undefined && ` (${positive ? "+" : ""}${deltaPct.toFixed(2)}%)`}
          </span>
          <span className="text-[10px] text-[var(--text-muted)]">{currency} change</span>
        </div>
      )}
    </motion.div>
  )
}
