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
            <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
              {fmtPct(data.expected_stress_amount / data.total_exposure)} of book
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
              <div key={d.grade} className="border border-border bg-bg rounded-sm p-3" data-testid={`grade-cell-${d.grade}`}>
                <GradeBadge grade={d.grade} size="sm" />
                <div className="font-mono text-lg mt-2">{d.accounts.toLocaleString("en-IN")}</div>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
                  {fmtInr(d.exposure)}
                </div>
              </div>
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
          <div key={r[keyField]} className="border border-border bg-bg rounded-sm p-3">
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
          </div>
        );
      })}
    </div>
  );
}
