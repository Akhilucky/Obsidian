import { motion } from "framer-motion"
import type { ReactNode } from "react"

type Props = {
  title?: string
  badge?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  hover?: boolean
  delay?: number
}

export default function Card({
  title,
  badge,
  actions,
  children,
  className = "",
  hover = false,
  delay = 0,
}: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut", delay }}
      className={`obs-card ${hover ? "obs-card-hover" : ""} ${className}`}
    >
      {(title || actions) && (
        <div className="flex items-center justify-between px-5 pt-4 pb-1">
          <div className="flex items-center gap-2">
            {title && (
              <span className="text-[12px] font-semibold tracking-[0.08em] uppercase text-[var(--text-secondary)]">
                {title}
              </span>
            )}
            {badge}
          </div>
          {actions}
        </div>
      )}
      <div className="p-5 pt-3">{children}</div>
    </motion.div>
  )
}
