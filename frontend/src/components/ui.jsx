import React from "react";
import { GRADE_META, cn } from "../lib/api";

export function Panel({ children, className, title, testId }) {
  return (
    <div
      data-testid={testId}
      className={cn(
        "border border-border bg-bg-surface rounded-sm",
        className
      )}
    >
      {title && (
        <div className="px-5 py-3 border-b border-border">
          <h3 className="text-xs uppercase tracking-widest2 text-fg-muted font-heading">
            {title}
          </h3>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}

export function Metric({ label, value, unit, testId, subtle }) {
  return (
    <div data-testid={testId}>
      <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
        {label}
      </div>
      <div
        className={cn(
          "font-mono tracking-tighter mt-2",
          subtle ? "text-2xl text-fg" : "text-4xl text-fg font-medium"
        )}
      >
        {value}
        {unit && <span className="text-fg-muted text-base ml-1">{unit}</span>}
      </div>
    </div>
  );
}

export function GradeBadge({ grade, size = "md", testId }) {
  const meta = GRADE_META[grade] || GRADE_META.Black;
  const sz =
    size === "lg"
      ? "text-sm px-3 py-1.5"
      : size === "sm"
      ? "text-[10px] px-1.5 py-0.5"
      : "text-xs px-2 py-1";
  return (
    <span
      data-testid={testId}
      className={cn("inline-flex items-center border rounded-sm font-mono uppercase tracking-widest2", sz)}
      style={{
        color: meta.color,
        borderColor: meta.color,
        backgroundColor: meta.bg,
      }}
    >
      {meta.label}
    </span>
  );
}

export function Btn({ children, onClick, variant = "primary", type = "button", disabled, testId, className }) {
  const base =
    "inline-flex items-center justify-center px-4 py-2 text-xs uppercase tracking-widest2 font-heading rounded-sm border transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary:
      "bg-primary text-white border-primary hover:bg-blue-500 hover:border-blue-500",
    ghost:
      "bg-transparent text-fg border-border hover:bg-bg-hover hover:border-border-strong",
    danger:
      "bg-transparent text-grade-red border-grade-red/50 hover:bg-grade-red/10",
  };
  return (
    <button
      data-testid={testId}
      onClick={onClick}
      disabled={disabled}
      type={type}
      className={cn(base, variants[variant], className)}
    >
      {children}
    </button>
  );
}

export function Loader({ label = "Loading" }) {
  return (
    <div className="flex items-center gap-3 text-fg-muted text-xs uppercase tracking-widest2">
      <span className="w-2 h-2 bg-primary animate-pulse" />
      {label}...
    </div>
  );
}

export function Divider() {
  return <div className="h-px bg-border w-full" />;
}
