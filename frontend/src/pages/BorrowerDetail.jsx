import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api, fmtInr, fmtPct, fmtNum, GRADE_META } from "../lib/api";
import { Panel, Metric, GradeBadge, Loader, Btn } from "../components/ui";
import { ArrowLeft, TrendingDown, TrendingUp, Shield, ClipboardList } from "lucide-react";
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from "recharts";

export default function BorrowerDetail() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    api
      .get(`/borrowers/${id}`)
      .then((r) => setData(r.data))
      .catch((e) => setErr(e.response?.data?.detail || e.message));
  }, [id]);

  if (err) return <div className="p-8 text-grade-red text-sm">Error: {err}</div>;
  if (!data) return <div className="p-8"><Loader label="Scoring borrower" /></div>;

  const gradeColor = GRADE_META[data.risk_grade]?.color || "#a1a1aa";

  return (
    <div className="p-8 space-y-6" data-testid="borrower-detail-page">
      <div className="flex items-center gap-4">
        <Link to="/borrowers" className="text-fg-muted hover:text-fg" data-testid="back-btn">
          <ArrowLeft size={18} />
        </Link>
        <div>
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading font-mono">
            {data.borrower_id} · {data.segment} · {data.loan_type}
          </div>
          <h1 className="text-2xl font-heading font-light tracking-tight mt-1" data-testid="borrower-name">
            {data.borrower_name}
          </h1>
          <div className="text-xs text-fg-muted mt-1">
            {data.sector} · {data.geography} · {data.constitution} · Vintage{" "}
            <span className="font-mono">{fmtNum(data.business_vintage_years, 1)}y</span>
          </div>
        </div>
        <div className="ml-auto">
          <GradeBadge grade={data.risk_grade} size="lg" testId="grade-badge-hero" />
        </div>
      </div>

      {/* Hero PD + Health */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <Panel className="lg:col-span-4" testId="pd-panel">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
            12-Month Probability of Default
          </div>
          <div className="flex items-baseline gap-3 mt-3">
            <div className="font-mono text-5xl tracking-tighter font-medium" style={{ color: gradeColor }}>
              {fmtPct(data.pd_12m, 2)}
            </div>
          </div>
          <div className="mt-4 h-2 bg-bg rounded-sm overflow-hidden">
            <div
              style={{
                width: `${Math.min(100, data.pd_12m * 100 * 3)}%`,
                background: gradeColor,
                height: "100%",
              }}
            />
          </div>
          <div className="flex justify-between text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-2">
            <span>0%</span><span>Green ≤5%</span><span>Red ≥20%</span><span>33%+</span>
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-fg-muted">
            <ClipboardList size={14} />
            Model v{data.model_version} · Data Quality
            <span className="font-mono text-fg ml-1">{data.data_quality_score}/100</span>
          </div>
        </Panel>

        <Panel className="lg:col-span-4" testId="health-panel">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
            MSME Health Score
          </div>
          <div className="relative mt-1">
            <ResponsiveContainer width="100%" height={180}>
              <RadialBarChart
                innerRadius="70%"
                outerRadius="100%"
                data={[{ v: data.health_score, fill: healthColor(data.health_score) }]}
                startAngle={220}
                endAngle={-40}
              >
                <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                <RadialBar background={{ fill: "#161616" }} dataKey="v" cornerRadius={0} />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <div className="font-mono text-5xl tracking-tighter font-medium">
                {fmtNum(data.health_score, 0)}
              </div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                {data.health_band}
              </div>
            </div>
          </div>
        </Panel>

        <Panel className="lg:col-span-4" testId="exposure-panel">
          <div className="grid grid-cols-2 gap-4">
            <Metric label="Sanctioned" value={fmtInr(data.sanctioned_limit)} subtle testId="m-sanction" />
            <Metric label="Outstanding" value={fmtInr(data.outstanding_amount)} subtle testId="m-outstanding" />
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                Expected Stress (12M)
              </div>
              <div className="font-mono text-2xl mt-2 text-grade-red">
                {fmtInr(data.expected_stress_amount)}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                CGTMSE
              </div>
              <div className="font-mono text-base mt-2 text-fg">
                {data.CGTMSE_flag || "No"}
              </div>
              {data.cgtmse_recommendation && (
                <div className="text-[10px] text-fg-muted mt-1 uppercase tracking-widest2 font-heading">
                  {data.cgtmse_recommendation}
                </div>
              )}
            </div>
          </div>
        </Panel>
      </div>

      {/* Recommended action */}
      <Panel title="Banker Action" testId="action-panel">
        <div className="flex items-start gap-3">
          <Shield size={18} className="mt-0.5" style={{ color: gradeColor }} />
          <div className="flex-1">
            <div className="text-fg" data-testid="action-narrative">{data.recommended_action}</div>
            <ul className="mt-3 space-y-1">
              {(data.action_checklist || []).map((a, i) => (
                <li key={i} className="text-sm text-fg-muted flex items-start gap-2">
                  <span className="font-mono text-fg-faint text-xs mt-0.5">▸</span>
                  {a}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Panel>

      {/* Reason codes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ReasonPanel
          title="Top Risk Drivers"
          items={data.top_risk_drivers}
          icon={TrendingDown}
          color="#EF4444"
          testId="risk-drivers"
        />
        <ReasonPanel
          title="Top Strength Drivers"
          items={data.top_strength_drivers}
          icon={TrendingUp}
          color="#10B981"
          testId="strength-drivers"
        />
      </div>

      {/* Sub scores */}
      {data.health_sub_scores && (
        <Panel title="Health Sub-Scores" testId="sub-scores">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(data.health_sub_scores).map(([k, v]) => (
              <div key={k}>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                  {k.replace(/_/g, " ")}
                </div>
                <div className="font-mono text-xl mt-1">{fmtNum(v)}</div>
                <div className="h-1 bg-bg mt-2">
                  <div style={{ width: `${v}%`, height: "100%", background: healthColor(v) }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* Signals */}
      <Panel title="Alternate & Structured Signals" testId="signals-panel">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          {Object.entries(data.signals || {}).map(([k, v]) => (
            <div key={k} className="border-b border-border pb-2">
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                {k.replace(/_/g, " ")}
              </div>
              <div className="font-mono mt-1 text-fg">
                {v === null || v === undefined ? "—" : typeof v === "number" ? fmtNum(v, 2) : String(v)}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Banker remarks */}
      <Panel title="Unstructured Banker Remarks" testId="remarks-panel">
        <div className="space-y-3">
          {Object.entries(data.remarks || {}).map(([k, v]) => (
            <div key={k} className="border-l-2 border-border pl-3">
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                {k.replace(/_/g, " ")}
              </div>
              <div className="text-sm text-fg mt-1">{v || "—"}</div>
            </div>
          ))}
        </div>
        <div className="mt-4">
          <Link to={`/notes?borrower=${data.borrower_id}`}>
            <Btn variant="ghost" testId="analyze-notes-btn">Run LLM NLP on These Notes →</Btn>
          </Link>
        </div>
      </Panel>
    </div>
  );
}

function ReasonPanel({ title, items, icon: Icon, color, testId }) {
  return (
    <div className="border border-border bg-bg-surface rounded-sm" data-testid={testId}>
      <div className="px-5 py-3 border-b border-border flex items-center gap-2">
        <Icon size={14} style={{ color }} />
        <h3 className="text-xs uppercase tracking-widest2 text-fg-muted font-heading">{title}</h3>
      </div>
      <div className="p-5 space-y-3">
        {(items || []).map((r, i) => (
          <div key={i} className="border border-border bg-bg rounded-sm p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <div className="font-mono text-xs" style={{ color }}>
                  {r.code}
                </div>
                <div className="text-sm text-fg mt-1">{r.description}</div>
              </div>
              <div className="font-mono text-xs text-fg-muted whitespace-nowrap">
                {r.impact >= 0 ? "+" : ""}
                {fmtNum(r.impact, 3)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function healthColor(v) {
  if (v >= 80) return "#10B981";
  if (v >= 65) return "#84CC16";
  if (v >= 50) return "#FBBF24";
  if (v >= 35) return "#F59E0B";
  return "#EF4444";
}
