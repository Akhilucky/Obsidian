type Props = {
  status: "up" | "down" | "neutral" | "live"
  size?: number
}

export default function StatusDot({ status, size = 6 }: Props) {
  const color =
    status === "up"
      ? "var(--up)"
      : status === "down"
        ? "var(--down)"
        : status === "live"
          ? "var(--accent)"
          : "var(--text-muted)"
  return (
    <span
      className={status === "live" ? "live-dot inline-block" : "inline-block"}
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: color,
        boxShadow: `0 0 8px ${color}66`,
      }}
    />
  )
}
