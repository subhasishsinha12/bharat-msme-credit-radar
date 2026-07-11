import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api, fmtInr, fmtPct, fmtNum } from "../lib/api";
import { Loader } from "../components/ui";
import { Printer, ArrowLeft } from "lucide-react";

export default function OfficerMemo() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  useEffect(() => {
    api.get(`/borrowers/${id}`).then((r) => setData(r.data));
    api.get(`/notes/history`, { params: { borrower_id: id, limit: 1 } }).then((r) => {
      if (r.data.items?.[0]) setAnalysis(r.data.items[0].analysis);
    });
  }, [id]);

  if (!data) return <div className="p-8"><Loader label="Preparing memo" /></div>;

  return (
    <div className="bg-white text-neutral-900 min-h-screen memo-root">
      <style>{`
        @media print {
          @page { size: A4; margin: 12mm; }
          .memo-noprint { display: none !important; }
          .memo-root { background: white !important; }
        }
        .memo-root { font-family: 'IBM Plex Sans', -apple-system, system-ui, sans-serif; }
        .memo-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
        .memo-heading { font-family: 'Chivo', sans-serif; letter-spacing: -0.01em; }
      `}</style>

      <div className="memo-noprint bg-neutral-100 border-b border-neutral-300 px-8 py-3 flex items-center justify-between">
        <Link to={`/borrowers/${id}`} className="text-neutral-600 hover:text-neutral-900 text-sm flex items-center gap-2">
          <ArrowLeft size={16} /> Back to borrower
        </Link>
        <button
          data-testid="print-memo-btn"
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 bg-neutral-900 text-white px-4 py-2 text-xs uppercase tracking-widest font-medium hover:bg-black"
        >
          <Printer size={14} /> Print / Save PDF
        </button>
      </div>

      <div className="max-w-4xl mx-auto p-10 space-y-6" data-testid="officer-memo">
        {/* Letterhead */}
        <div className="flex items-start justify-between border-b-2 border-neutral-900 pb-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading">
              Confidential · Credit Decision Support
            </div>
            <h1 className="text-3xl font-medium memo-heading mt-1">Officer Memo</h1>
            <div className="text-sm text-neutral-600 mt-1">Bharat MSME Credit Radar · 12-Month PD Engine</div>
          </div>
          <div className="text-right text-xs text-neutral-600 memo-mono">
            <div>Model {data.model_version}</div>
            <div>Generated {new Date().toLocaleString()}</div>
            <div>Data Quality {data.data_quality_score}/100</div>
          </div>
        </div>

        {/* Borrower */}
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading">Borrower</div>
            <div className="text-2xl memo-heading mt-1">{data.borrower_name}</div>
            <div className="memo-mono text-sm text-neutral-600 mt-1">{data.borrower_id}</div>
            <div className="text-sm text-neutral-700 mt-2">
              {data.segment} · {data.sector} · {data.geography} · {data.constitution} · Vintage{" "}
              <span className="memo-mono">{fmtNum(data.business_vintage_years, 1)}y</span>
            </div>
            <div className="text-sm text-neutral-700 mt-1">
              Facility: {data.loan_type} · CGTMSE: {data.CGTMSE_flag || "No"}
            </div>
          </div>
          <div className="border border-neutral-300 p-4">
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading">Early Warning Grade</div>
            <div className="text-3xl memo-heading mt-1" style={{ color: gradeColor(data.risk_grade) }}>
              {data.risk_grade}
            </div>
            <div className="text-xs text-neutral-600 mt-2 memo-mono">{data.health_band}</div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-4 border-t border-b border-neutral-300 py-5">
          <MetricBlock label="12M PD" value={fmtPct(data.pd_12m, 2)} color={gradeColor(data.risk_grade)} />
          <MetricBlock label="Health Score" value={`${fmtNum(data.health_score, 0)}/100`} />
          <MetricBlock label="Sanctioned" value={fmtInr(data.sanctioned_limit)} />
          <MetricBlock label="Outstanding" value={fmtInr(data.outstanding_amount)} />
        </div>

        {/* Action */}
        <div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading">Recommended Banker Action</div>
          <div className="text-base text-neutral-900 mt-1">{data.recommended_action}</div>
          <ul className="mt-2 space-y-1 text-sm text-neutral-700">
            {(data.action_checklist || []).map((a, i) => (
              <li key={i} className="flex gap-2">
                <span className="memo-mono text-neutral-400">▸</span>
                {a}
              </li>
            ))}
          </ul>
          {data.cgtmse_recommendation && (
            <div className="text-sm text-neutral-700 mt-2">
              <span className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading mr-1">CGTMSE:</span>
              {data.cgtmse_recommendation}
            </div>
          )}
        </div>

        {/* Reasons */}
        <div className="grid grid-cols-2 gap-6">
          <ReasonList title="Top Risk Drivers" items={data.top_risk_drivers} sign="+" color="#B91C1C" />
          <ReasonList title="Top Strength Drivers" items={data.top_strength_drivers} sign="−" color="#047857" />
        </div>

        {/* Sub scores */}
        {data.health_sub_scores && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading mb-2">Health Sub-Scores</div>
            <div className="grid grid-cols-4 gap-3 text-sm">
              {Object.entries(data.health_sub_scores).map(([k, v]) => (
                <div key={k} className="border border-neutral-200 p-2">
                  <div className="text-[10px] uppercase tracking-[0.15em] text-neutral-500 memo-heading">
                    {k.replace(/_/g, " ")}
                  </div>
                  <div className="memo-mono mt-1">{fmtNum(v)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Banker remarks */}
        {data.remarks && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading mb-2">Banker Remarks (Verbatim)</div>
            <div className="space-y-2 text-sm">
              {Object.entries(data.remarks).map(([k, v]) => (
                v ? (
                  <div key={k}>
                    <span className="text-[10px] uppercase tracking-[0.15em] text-neutral-500 memo-heading">
                      {k.replace(/_/g, " ")}:{" "}
                    </span>
                    <span className="text-neutral-800">{v}</span>
                  </div>
                ) : null
              ))}
            </div>
          </div>
        )}

        {/* NLP */}
        {analysis && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading mb-2">NLP Extracted Intelligence</div>
            <div className="text-sm text-neutral-800">
              Overall stress score <span className="memo-mono">{analysis.overall_stress_score}/100</span> · sentiment{" "}
              <span className="capitalize">{analysis.overall_sentiment}</span>
            </div>
            {analysis.summary && <div className="text-sm text-neutral-700 mt-1 italic">"{analysis.summary}"</div>}
          </div>
        )}

        {/* Footer */}
        <div className="border-t-2 border-neutral-900 pt-3 text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading flex justify-between">
          <span>Confidential · IDBI Innovate 2026 · Track 04 Prototype</span>
          <span>Not to be used for lending decisions without validation</span>
        </div>
      </div>
    </div>
  );
}

function MetricBlock({ label, value, color }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading">{label}</div>
      <div className="text-2xl memo-mono mt-1" style={{ color: color || undefined }}>{value}</div>
    </div>
  );
}

function ReasonList({ title, items, sign, color }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500 memo-heading mb-2">{title}</div>
      <div className="space-y-2">
        {(items || []).map((r, i) => (
          <div key={i} className="border-l-2 pl-3" style={{ borderColor: color }}>
            <div className="memo-mono text-xs" style={{ color }}>
              {r.code} <span className="text-neutral-500">{sign}{Math.abs(r.impact).toFixed(3)}</span>
            </div>
            <div className="text-sm text-neutral-800">{r.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function gradeColor(g) {
  return { Green: "#047857", Yellow: "#CA8A04", Amber: "#B45309", Red: "#B91C1C", Black: "#1F2937" }[g] || "#1F2937";
}
