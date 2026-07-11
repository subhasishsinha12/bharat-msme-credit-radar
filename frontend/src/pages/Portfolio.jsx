import React, { useEffect, useState } from "react";
import { api, fmtInr, fmtPct, fmtNum, GRADE_META } from "../lib/api";
import { Panel, Metric, GradeBadge, Loader } from "../components/ui";
import {
  BarChart,
  Bar,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import { Link } from "react-router-dom";

import { Activity, AlertOctagon, Sparkles } from "lucide-react";

const GRADE_ORDER = ["Green", "Yellow", "Amber", "Red", "Black"];

export default function Portfolio() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api
      .get("/portfolio/summary")
      .then((r) => setData(r.data))
      .catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="p-8 text-grade-red text-sm">Error: {err}</div>;
  if (!data) return <div className="p-8"><Loader label="Loading portfolio" /></div>;

  const gradeData = GRADE_ORDER.map((g) => ({
    grade: g,
    accounts: data.accounts_by_grade[g] || 0,
    exposure: data.exposure_by_grade[g] || 0,
  }));

  return (
    <div className="p-8 space-y-6" data-testid="portfolio-page">
      {/* Hero */}
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Portfolio Command Center
        </div>
        <h1 className="text-3xl sm:text-4xl font-heading font-light tracking-tight mt-1">
          MSME Credit Radar
        </h1>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Panel testId="kpi-accounts">
          <Metric
            label="Total Accounts"
            value={data.total_accounts.toLocaleString("en-IN")}
          />
        </Panel>
        <Panel testId="kpi-exposure">
          <Metric label="Total Exposure" value={fmtInr(data.total_exposure)} />
        </Panel>
        <Panel testId="kpi-stress">
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
              12M Expected Stress
            </div>
            <div className="font-mono tracking-tighter mt-2 text-4xl font-medium text-grade-red">
              {fmtInr(data.expected_stress_amount)}
            </div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1 flex items-center gap-1">
              {data.stress_mom_delta > 0 ? "▲" : "▼"}{" "}
              <span className={data.stress_mom_delta > 0 ? "text-grade-red" : "text-grade-green"}>
                {fmtInr(Math.abs(data.stress_mom_delta))} MoM
              </span>
              <span className="text-fg-faint">· {fmtPct(data.expected_stress_amount / data.total_exposure)} of book</span>
            </div>
          </div>
        </Panel>
        <Panel testId="kpi-avg-health">
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
              Avg Health Score
            </div>
            <div className="font-mono tracking-tighter mt-2 text-4xl font-medium">
              {fmtNum(data.average_health_score, 1)}
              <span className="text-fg-muted text-base ml-1">/100</span>
            </div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
              Avg PD {fmtPct(data.average_pd, 2)}
            </div>
          </div>
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Risk grade distribution */}
        <Panel title="Risk Grade Distribution" testId="risk-grade-dist" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={gradeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <XAxis
                dataKey="grade"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "JetBrains Mono" }}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "JetBrains Mono" }}
              />
              <Tooltip
                contentStyle={{
                  background: "#0A0A0A",
                  border: "1px solid #27272A",
                  borderRadius: 2,
                  fontSize: 12,
                  fontFamily: "JetBrains Mono",
                }}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
              />
              <Bar dataKey="accounts" radius={[0, 0, 0, 0]}>
                {gradeData.map((d) => (
                  <Cell key={d.grade} fill={GRADE_META[d.grade].color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-5 gap-3 mt-4">
            {gradeData.map((d) => (
              <Link
                key={d.grade}
                to={`/borrowers?grade=${d.grade}`}
                className="border border-border bg-bg rounded-sm p-3 hover:bg-bg-hover hover:border-border-strong transition-colors"
                data-testid={`grade-cell-${d.grade}`}
              >
                <GradeBadge grade={d.grade} size="sm" />
                <div className="font-mono text-lg mt-2">{d.accounts.toLocaleString("en-IN")}</div>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
                  {fmtInr(d.exposure)}
                </div>
              </Link>
            ))}
          </div>
        </Panel>

        {/* Exposure share pie */}
        <Panel title="Exposure Share by Grade" testId="exposure-share">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={gradeData.filter((d) => d.exposure > 0)}
                dataKey="exposure"
                nameKey="grade"
                innerRadius={55}
                outerRadius={85}
                paddingAngle={1}
                stroke="none"
              >
                {gradeData.map((d) => (
                  <Cell key={d.grade} fill={GRADE_META[d.grade].color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#0A0A0A",
                  border: "1px solid #27272A",
                  fontSize: 12,
                  fontFamily: "JetBrains Mono",
                }}
                formatter={(v) => fmtInr(v)}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5 mt-2">
            {gradeData.map((d) => (
              <div key={d.grade} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2" style={{ background: GRADE_META[d.grade].color }} />
                  <span className="text-fg-muted">{d.grade}</span>
                </div>
                <span className="font-mono text-fg">
                  {fmtPct(d.exposure / data.total_exposure)}
                </span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* Action Queue + Cluster Alerts + Growth Radar tile */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="Banker Action Queue" testId="action-queue-panel" className="lg:col-span-2">
          <div className="grid grid-cols-3 gap-3">
            {[
              { k: "field_visits", label: "Field Visits", color: "#F59E0B" },
              { k: "stock_audits", label: "Stock Audits", color: "#EF4444" },
              { k: "gst_bank_recon_reviews", label: "GST-Bank Recon", color: "#818CF8" },
              { k: "enhancement_freezes", label: "Enhancement Freeze", color: "#FBBF24" },
              { k: "urgent_recoveries", label: "Urgent Recovery", color: "#a1a1aa" },
              { k: "early_engagement_watchlist", label: "Early Engagement", color: "#10B981" },
            ].map((it) => (
              <Link
                key={it.k}
                to="/borrowers?grade=Red"
                className="border border-border bg-bg rounded-sm p-3 hover:bg-bg-hover transition-colors"
                data-testid={`aq-${it.k}`}
              >
                <div className="text-[10px] uppercase tracking-widest2 font-heading" style={{ color: it.color }}>
                  {it.label}
                </div>
                <div className="font-mono text-3xl mt-1">
                  {(data.action_queue?.[it.k] || 0).toLocaleString("en-IN")}
                </div>
              </Link>
            ))}
          </div>
        </Panel>

        <Panel title="Growth Radar" testId="growth-tile">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                Pre-Qualified Pipeline
              </div>
              <div className="font-mono text-4xl mt-2 font-medium text-primary">
                {fmtInr(data.growth_radar_tile?.pipeline || 0)}
              </div>
              <div className="text-sm text-fg-muted mt-1">
                <span className="font-mono text-grade-green">{data.growth_radar_tile?.candidates || 0}</span> healthy accounts
                flagged for WC enhancement or new term loan
              </div>
            </div>
            <Sparkles size={18} className="text-primary mt-1" />
          </div>
          <Link
            to="/growth"
            className="mt-4 inline-flex items-center gap-2 text-xs uppercase tracking-widest2 font-heading text-primary hover:text-blue-400"
            data-testid="growth-tile-link"
          >
            Open Growth Radar →
          </Link>
        </Panel>
      </div>

      {/* Cluster contagion alerts */}
      {(data.cluster_alerts || []).length > 0 && (
        <Panel title="Cluster Contagion Alerts" testId="cluster-alerts">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.cluster_alerts.map((a, i) => (
              <Link
                key={i}
                to={`/borrowers?sector=${encodeURIComponent(a.sector)}&geography=${encodeURIComponent(a.geography)}`}
                className="border border-grade-amber/40 bg-grade-amber/5 rounded-sm p-3 hover:border-grade-amber transition-colors"
                data-testid={`cluster-alert-${i}`}
              >
                <div className="flex items-start gap-2">
                  <AlertOctagon size={14} className="text-grade-amber mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-fg">{a.message}</div>
                    <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading mt-2 flex gap-3">
                      <span>Exposure: <span className="font-mono text-fg">{fmtInr(a.exposure)}</span></span>
                      <span>Expected Stress: <span className="font-mono text-grade-red">{fmtInr(a.expected_stress)}</span></span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </Panel>
      )}

      {/* Sector & geography heatmaps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Sector Risk Heatmap" testId="sector-heatmap">
          <SectorGeoTable rows={data.sector_wise_summary} keyField="sector" />
        </Panel>
        <Panel title="Geography Risk Heatmap" testId="geo-heatmap">
          <SectorGeoTable rows={data.geography_wise_summary} keyField="geography" />
        </Panel>
      </div>

      {/* Top 10 high risk */}
      <Panel title="Top 10 High-Risk Accounts · Action Queue" testId="top-10-risk">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading border-b border-border">
                <th className="text-left py-2 pr-3">Borrower</th>
                <th className="text-left py-2 pr-3">Segment</th>
                <th className="text-left py-2 pr-3">Sector</th>
                <th className="text-left py-2 pr-3">Geo</th>
                <th className="text-right py-2 pr-3">Outstanding</th>
                <th className="text-right py-2 pr-3">PD 12M</th>
                <th className="text-right py-2 pr-3">Health</th>
                <th className="text-left py-2 pl-3">Grade</th>
              </tr>
            </thead>
            <tbody>
              {data.top_10_high_risk_accounts.map((b) => (
                <tr key={b.borrower_id} className="border-b border-border hover:bg-bg-hover transition-colors">
                  <td className="py-3 pr-3">
                    <Link
                      to={`/borrowers/${b.borrower_id}`}
                      className="text-fg hover:text-primary"
                      data-testid={`top-risk-link-${b.borrower_id}`}
                    >
                      <div className="font-mono text-xs">{b.borrower_id}</div>
                      <div className="text-fg-muted text-xs">{b.borrower_name}</div>
                    </Link>
                  </td>
                  <td className="py-3 pr-3 text-fg-muted text-xs">{b.segment}</td>
                  <td className="py-3 pr-3 text-fg-muted text-xs">{b.sector}</td>
                  <td className="py-3 pr-3 text-fg-muted text-xs">{b.geography}</td>
                  <td className="py-3 pr-3 text-right font-mono">{fmtInr(b.outstanding_amount)}</td>
                  <td className="py-3 pr-3 text-right font-mono text-grade-red">
                    {fmtPct(b.pd_12m, 1)}
                  </td>
                  <td className="py-3 pr-3 text-right font-mono">{fmtNum(b.health_score)}</td>
                  <td className="py-3 pl-3"><GradeBadge grade={b.risk_grade} size="sm" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function SectorGeoTable({ rows, keyField }) {
  const max = Math.max(...rows.map((r) => r.expected_stress_amount || 0), 1);
  const paramKey = keyField === "sector" ? "sector" : "geography";
  return (
    <div className="space-y-2">
      {rows.slice(0, 10).map((r) => {
        const pct = ((r.expected_stress_amount || 0) / max) * 100;
        const pdPct = (r.avg_pd || 0) * 100;
        const color =
          pdPct > 20
            ? "#EF4444"
            : pdPct > 10
            ? "#F59E0B"
            : pdPct > 5
            ? "#FBBF24"
            : "#10B981";
        return (
          <Link
            key={r[keyField]}
            to={`/borrowers?${paramKey}=${encodeURIComponent(r[keyField])}`}
            className="block border border-border bg-bg rounded-sm p-3 hover:bg-bg-hover hover:border-border-strong transition-colors"
            data-testid={`heatmap-${keyField}-${r[keyField]}`}
          >
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-fg">{r[keyField]}</span>
              <span className="font-mono text-fg-muted">
                {r.accounts} accts · {fmtInr(r.total_exposure)}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-1.5 bg-bg-panel">
                <div style={{ width: `${pct}%`, background: color, height: "100%" }} />
              </div>
              <span className="font-mono text-xs w-16 text-right" style={{ color }}>
                {fmtPct(r.avg_pd, 1)}
              </span>
              <span className="font-mono text-xs text-fg-muted w-20 text-right">
                {fmtInr(r.expected_stress_amount)}
              </span>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
