import React, { useEffect, useState } from "react";
import { api, fmtNum, fmtPct } from "../lib/api";
import { Panel, Loader } from "./ui";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";

const TOOLTIP_STYLE = {
  background: "#0A0A0A",
  border: "1px solid #27272A",
  borderRadius: 2,
  fontSize: 12,
  fontFamily: "JetBrains Mono",
};

export default function TrajectoryPanel({ borrowerId }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    setData(null);
    api.get(`/borrowers/${borrowerId}/history`).then((r) => setData(r.data.trajectory));
  }, [borrowerId]);

  if (!data) return <Panel title="8-Month Trajectory"><Loader label="Scoring historical panel" /></Panel>;

  const series = data.map((d) => ({
    month: `M${d.obs_month}`,
    pd: +(d.pd_12m * 100).toFixed(2),
    health: +d.health_score.toFixed(1),
    turnover: +(d.gst_turnover_12m / 1e5).toFixed(1),
    cc: +d.cc_utilization_avg_3m.toFixed(1),
    dpd: d.current_dpd,
    bureau: d.bureau_score,
  }));

  return (
    <Panel title="8-Month Trajectory · MSME Credit Twin" testId="trajectory-panel">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MiniChart title="12-Month PD (%)" data={series} dataKey="pd" color="#EF4444" refLines={[{ y: 5, color: "#10B981" }, { y: 20, color: "#F59E0B" }]} formatter={(v) => `${v.toFixed(2)}%`} />
        <MiniChart title="MSME Health Score" data={series} dataKey="health" color="#10B981" formatter={(v) => v.toFixed(0)} yDomain={[0, 100]} />
        <MiniChart title="GST Turnover (₹ L, 12M)" data={series} dataKey="turnover" color="#2563EB" formatter={(v) => `₹${v.toFixed(1)} L`} />
        <MiniChart title="Current DPD (days)" data={series} dataKey="dpd" color="#F59E0B" formatter={(v) => `${v.toFixed(0)}d`} />
        <MiniChart title="CC Utilization %" data={series} dataKey="cc" color="#FBBF24" formatter={(v) => `${v.toFixed(1)}%`} refLines={[{ y: 90, color: "#EF4444" }]} yDomain={[0, 120]} />
        <MiniChart title="Bureau Score" data={series} dataKey="bureau" color="#818CF8" formatter={(v) => v.toFixed(0)} yDomain={[300, 900]} />
      </div>
    </Panel>
  );
}

function MiniChart({ title, data, dataKey, color, formatter, refLines = [], yDomain }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading mb-2">
        {title}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="month"
            axisLine={false}
            tickLine={false}
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: "#475569", fontSize: 10, fontFamily: "JetBrains Mono" }}
            width={38}
            domain={yDomain || ["auto", "auto"]}
          />
          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatter ? formatter(v) : v} labelStyle={{ color: "#94A3B8" }} />
          {refLines.map((r, i) => (
            <ReferenceLine key={i} y={r.y} stroke={r.color} strokeDasharray="3 3" strokeOpacity={0.6} />
          ))}
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            dot={{ fill: color, r: 3, strokeWidth: 0 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
