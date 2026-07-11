import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api, fmtInr, fmtPct, fmtNum, GRADE_META } from "../lib/api";
import { Panel, Metric, GradeBadge, Loader, Btn } from "../components/ui";
import TrajectoryPanel from "../components/TrajectoryPanel";
import { ArrowLeft, TrendingDown, TrendingUp, Shield, ClipboardList, Printer } from "lucide-react";
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
        <div className="ml-auto flex items-center gap-3">
          <Link to={`/borrowers/${data.borrower_id}/memo`}>
            <Btn variant="ghost" testId="print-memo-btn">
              <Printer size={14} className="mr-2" /> Officer Memo
            </Btn>
          </Link>
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

      {/* SMA migration + Confidence + Segment Benchmark row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="SMA Migration Probability" testId="sma-panel">
          <div className="space-y-3">
            {[
              { k: "sma_0_to_1", label: "SMA-0 → SMA-1", desc: "Overdue 0-30 days" },
              { k: "sma_1_to_2", label: "SMA-1 → SMA-2", desc: "Overdue 30-60 days" },
              { k: "sma_2_to_npa", label: "SMA-2 → NPA", desc: "Overdue 60-90 days" },
            ].map((it) => {
              const v = data.sma_migration?.[it.k] || 0;
              const color = v > 0.5 ? "#EF4444" : v > 0.2 ? "#F59E0B" : "#10B981";
              return (
                <div key={it.k}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-fg-muted">{it.label}</span>
                    <span className="font-mono" style={{ color }}>{fmtPct(v, 1)}</span>
                  </div>
                  <div className="h-1 bg-bg mt-1">
                    <div style={{ width: `${v * 100}%`, background: color, height: "100%" }} />
                  </div>
                  <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">{it.desc}</div>
                </div>
              );
            })}
          </div>
        </Panel>

        <Panel title="Model Confidence" testId="confidence-panel">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Confidence Score</div>
          <div className="font-mono text-5xl tracking-tighter mt-2" style={{ color: (data.confidence_score || 0) >= 75 ? "#10B981" : (data.confidence_score || 0) >= 60 ? "#F59E0B" : "#EF4444" }}>
            {data.confidence_score}%
          </div>
          <div className="h-2 bg-bg mt-3">
            <div style={{ width: `${data.confidence_score || 0}%`, height: "100%", background: (data.confidence_score || 0) >= 75 ? "#10B981" : "#F59E0B" }} />
          </div>
          <div className="text-xs text-fg-muted mt-3">
            Combines data quality ({data.data_quality_score}/100) and SHAP-driver separation for this borrower.
          </div>
        </Panel>

        <Panel title="Segment Benchmark" testId="segment-benchmark">
          {data.segment_benchmark ? (
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
                vs {data.segment_benchmark.segment} peers ({data.segment_benchmark.peer_count.toLocaleString("en-IN")})
              </div>
              <div className="mt-3 space-y-3">
                <PctBar
                  label="PD Percentile"
                  value={data.segment_benchmark.percentile_pd}
                  higherIsWorse={true}
                  note={`Borrower ${fmtPct(data.segment_benchmark.borrower_pd, 1)} · Peer avg ${fmtPct(data.segment_benchmark.peer_avg_pd, 1)}`}
                />
                <PctBar
                  label="Health Percentile"
                  value={data.segment_benchmark.percentile_health}
                  higherIsWorse={false}
                />
              </div>
            </div>
          ) : (
            <div className="text-xs text-fg-muted">Segment benchmark unavailable</div>
          )}
        </Panel>
      </div>

      {/* Data Trust + GST Authenticity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Data Trust · Fraud Intelligence" testId="data-trust-panel">
          {data.data_trust && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Data Quality</div>
                  <div className="font-mono text-3xl mt-1 text-grade-green">{data.data_trust.data_quality_score}/100</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Fraud Score</div>
                  <div className="font-mono text-3xl mt-1" style={{ color: data.data_trust.fraud_score > 50 ? "#EF4444" : data.data_trust.fraud_score > 25 ? "#F59E0B" : "#10B981" }}>
                    {data.data_trust.fraud_score}/100
                  </div>
                </div>
              </div>
              <div className="mt-4 space-y-2">
                {Object.entries(data.data_trust.components || {}).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between text-xs">
                    <span className="text-fg-muted uppercase tracking-widest2 font-heading">{k.replace(/_/g, " ")}</span>
                    <span className="font-mono" style={{ color: v > 0 ? "#EF4444" : "#10B981" }}>
                      {v > 0 ? "FLAGGED" : "CLEAN"}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </Panel>

        <Panel title="GST Authenticity Engine" testId="gst-authenticity-panel">
          {data.gst_authenticity && (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <GstCell label="GST Status" value={data.gst_authenticity.gst_status} />
              <GstCell label="Registration Age" value={`${fmtNum(data.gst_authenticity.gst_registration_age_years, 1)}y`} />
              <GstCell label="Filing Delays 6M" value={fmtNum(data.gst_authenticity.gst_filing_delay_count_6m, 0)} risk={data.gst_authenticity.gst_filing_delay_count_6m > 2} />
              <GstCell label="GSTR1 vs 3B Mismatch" value={`${fmtNum(data.gst_authenticity.gstr1_vs_3b_mismatch_pct, 1)}%`} risk={data.gst_authenticity.gstr1_vs_3b_mismatch_pct > 15} />
              <GstCell label="ITC-to-Sales" value={fmtNum(data.gst_authenticity.itc_to_sales_ratio, 3)} risk={data.gst_authenticity.itc_to_sales_ratio > 0.9} />
              <GstCell label="E-way Bill Mismatch" value={data.gst_authenticity.eway_bill_mismatch_flag > 0 ? "FLAGGED" : "Clean"} risk={data.gst_authenticity.eway_bill_mismatch_flag > 0} />
              <GstCell label="Nil Returns 12M" value={fmtNum(data.gst_authenticity.nil_return_count_12m, 0)} risk={data.gst_authenticity.nil_return_count_12m > 0} />
              <GstCell label="Sudden Turnover Spike" value={data.gst_authenticity.sudden_turnover_spike_flag > 0 ? "FLAGGED" : "Clean"} risk={data.gst_authenticity.sudden_turnover_spike_flag > 0} />
              <GstCell label="Top-2 Buyers %" value={`${fmtNum(data.gst_authenticity.buyer_concentration_top2_pct, 0)}%`} risk={data.gst_authenticity.buyer_concentration_top2_pct > 60} />
              <GstCell label="Top-2 Suppliers %" value={`${fmtNum(data.gst_authenticity.supplier_concentration_top2_pct, 0)}%`} risk={data.gst_authenticity.supplier_concentration_top2_pct > 60} />
            </div>
          )}
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

      {/* Trajectory (MSME Credit Twin) */}
      <TrajectoryPanel borrowerId={data.borrower_id} />

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

function PctBar({ label, value, higherIsWorse, note }) {
  const v = Number(value) || 0;
  const bad = higherIsWorse ? v > 70 : v < 30;
  const mid = higherIsWorse ? v > 40 : v < 60;
  const color = bad ? "#EF4444" : mid ? "#F59E0B" : "#10B981";
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-fg-muted">{label}</span>
        <span className="font-mono" style={{ color }}>{v.toFixed(0)}%</span>
      </div>
      <div className="h-1 bg-bg mt-1">
        <div style={{ width: `${v}%`, background: color, height: "100%" }} />
      </div>
      {note && <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">{note}</div>}
    </div>
  );
}

function GstCell({ label, value, risk }) {
  return (
    <div className="border border-border bg-bg rounded-sm p-2">
      <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">{label}</div>
      <div className="font-mono mt-1" style={{ color: risk ? "#EF4444" : "#F8FAFC" }}>{value ?? "—"}</div>
    </div>
  );
}
