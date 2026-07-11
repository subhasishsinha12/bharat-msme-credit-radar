import React, { useEffect, useState } from "react";
import { api, fmtNum, fmtPct } from "../lib/api";
import { Panel, Loader } from "../components/ui";
import {
  BarChart, Bar, ResponsiveContainer, XAxis, YAxis, Tooltip, LineChart, Line, ReferenceLine,
} from "recharts";
import { Activity, Target, TrendingUp, ShieldCheck } from "lucide-react";

const TOOLTIP_STYLE = {
  background: "#0A0A0A", border: "1px solid #27272A", fontSize: 12,
  fontFamily: "JetBrains Mono", borderRadius: 2,
};

export default function ModelPerformance() {
  const [data, setData] = useState(null);
  useEffect(() => { api.get("/model/performance").then((r) => setData(r.data)); }, []);

  if (!data) return <div className="p-8"><Loader label="Loading model performance" /></div>;
  const m = data.calibrated;

  const kpis = [
    { label: "AUC-ROC", value: fmtNum(m.auc_roc, 3), sub: "Strong rank-ordering (deck target 0.87)", icon: Activity, color: "#10B981" },
    { label: "Gini", value: fmtNum(m.gini, 3), sub: "Regulator-familiar discrimination", icon: Target, color: "#10B981" },
    { label: "KS Statistic", value: fmtNum(m.ks_statistic * 100, 1), sub: "Good vs. bad separation", icon: Activity, color: "#10B981" },
    { label: "Recall @ Top-20%", value: `${(m.recall_at_top20pct * 100).toFixed(1)}%`, sub: "vs 16-22% baseline today", icon: TrendingUp, color: "#10B981" },
    { label: "Top-Decile Lift", value: `${fmtNum(m.top_decile_lift, 2)}×`, sub: "Riskiest 10% concentration", icon: TrendingUp, color: "#818CF8" },
    { label: "Brier Score", value: fmtNum(m.brier_score, 3), sub: "Calibrated probabilities", icon: ShieldCheck, color: "#22D3EE" },
    { label: "PSI Dev vs Holdout", value: fmtNum(data.psi_dev_vs_holdout, 3), sub: "No population drift", icon: ShieldCheck, color: "#22D3EE" },
    { label: "Precision · Recall", value: `${(m.precision * 100).toFixed(0)}% · ${(m.recall * 100).toFixed(0)}%`, sub: `F1 ${fmtNum(m.f1_score, 3)}`, icon: Target },
  ];

  // Calibration curve
  const calCurve = (data.calibration_curve || []).map((p) => ({
    predicted: +(p.mean_predicted_pd * 100).toFixed(2),
    observed: +(p.observed_stress_rate * 100).toFixed(2),
    n: p.n,
  }));

  const cm = data.confusion_matrix_calibrated || [[0, 0], [0, 0]];

  return (
    <div className="p-8 space-y-6" data-testid="performance-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Prototype Performance · Benchmarking
        </div>
        <h1 className="text-3xl font-heading font-light tracking-tight mt-1">Model Performance</h1>
        <p className="text-sm text-fg-muted mt-2 max-w-3xl">
          Metrics from the calibrated <span className="font-mono text-fg">{data.best_model_name}</span> ensemble on {data.training_universe.borrower_months.toLocaleString("en-IN")} borrower-months
          ({data.training_universe.borrowers.toLocaleString("en-IN")} borrowers × {data.training_universe.months_per_borrower} months, stress rate {fmtPct(data.training_universe.stress_rate, 1)}).
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {kpis.map((k, i) => (
          <Panel key={i} testId={`perf-kpi-${i}`}>
            <div className="flex items-start justify-between">
              <div>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">{k.label}</div>
                <div className="font-mono text-3xl mt-2 font-medium" style={{ color: k.color || "#F8FAFC" }}>{k.value}</div>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">{k.sub}</div>
              </div>
              <k.icon size={16} className="text-fg-faint mt-1" />
            </div>
          </Panel>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="Reliability / Calibration Curve" testId="calibration-panel" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={calCurve} margin={{ top: 10, right: 15, left: 0, bottom: 0 }}>
              <XAxis dataKey="predicted" tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} label={{ value: "Predicted PD (%)", position: "insideBottom", offset: -5, fill: "#475569", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} label={{ value: "Observed Stress (%)", angle: -90, position: "insideLeft", fill: "#475569", fontSize: 10 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => `${v}%`} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="#3F3F46" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="observed" stroke="#10B981" strokeWidth={2.5} dot={{ r: 4, fill: "#10B981" }} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-3">
            Solid line = observed stress · dashed = perfect calibration
          </div>
        </Panel>

        <Panel title="Confusion Matrix (Calibrated)" testId="cm-panel">
          <div className="grid grid-cols-2 gap-2">
            <CmCell label="TN" value={cm[0][0]} color="#10B981" />
            <CmCell label="FP" value={cm[0][1]} color="#F59E0B" />
            <CmCell label="FN" value={cm[1][0]} color="#EF4444" />
            <CmCell label="TP" value={cm[1][1]} color="#10B981" />
          </div>
          <div className="grid grid-cols-2 text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-3">
            <div>Actual → Predicted ↓</div>
            <div className="text-right">n = {m.n.toLocaleString("en-IN")}</div>
          </div>
        </Panel>
      </div>

      <Panel title="Interpretation" testId="perf-interpretation">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-fg-muted">
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg font-heading">Recall @ Top-20% Risk Band</div>
            <div className="mt-1">
              The engine catches <span className="font-mono text-grade-green">{(m.recall_at_top20pct * 100).toFixed(1)}%</span> of future stress
              accounts in its Red/Amber bands — versus the <span className="font-mono">16-22%</span> baseline seen with today's document-based
              MSME workflows.
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg font-heading">Top-Decile Lift</div>
            <div className="mt-1">
              The riskiest <span className="font-mono">10%</span> of accounts concentrate
              <span className="font-mono text-primary"> {fmtNum(m.top_decile_lift, 2)}×</span> the base stress rate — directly enables
              triaged field visits and stock audits by the branch.
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg font-heading">Brier Score</div>
            <div className="mt-1">
              At <span className="font-mono">{fmtNum(m.brier_score, 3)}</span> the model is well-calibrated —
              a predicted PD of 18% corresponds to ~18% observed stress in that bucket.
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest2 text-fg font-heading">PSI Dev vs Holdout</div>
            <div className="mt-1">
              PSI of <span className="font-mono">{fmtNum(data.psi_dev_vs_holdout, 3)}</span> indicates a stable population and no
              distributional drift between development and holdout samples.
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function CmCell({ label, value, color }) {
  return (
    <div className="border border-border bg-bg rounded-sm p-3">
      <div className="text-[10px] uppercase tracking-widest2 font-heading" style={{ color }}>{label}</div>
      <div className="font-mono text-2xl mt-1">{value.toLocaleString("en-IN")}</div>
    </div>
  );
}
