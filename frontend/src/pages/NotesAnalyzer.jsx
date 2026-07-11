import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import { Panel, Btn, Loader, GradeBadge } from "../components/ui";
import { toast, Toaster } from "sonner";
import { Sparkles, AlertTriangle } from "lucide-react";

const SAMPLE = {
  cam_remarks:
    "Turnover declining YoY; promoter mentions delayed payments from two large buyers. Working capital cycle stretched to 90 days.",
  fi_remarks:
    "Field visit confirmed premises operational but 40% of machinery idle. Signboard faded, staff strength down from 18 to 11.",
  rcu_remarks:
    "Two invoices could not be verified with counterparties. GST filing 45 days delayed.",
  collection_remarks:
    "EMI bounced twice in last quarter; promise-to-pay not honoured last cycle.",
  stock_inspection_remarks:
    "Physical stock 22% below book value; slow-moving inventory beyond 180 days.",
};

const SEV = {
  low: "#FBBF24",
  medium: "#F59E0B",
  high: "#EF4444",
};

export default function NotesAnalyzer() {
  const [sp] = useSearchParams();
  const borrower = sp.get("borrower") || "";
  const [notes, setNotes] = useState({
    borrower_id: borrower,
    cam_remarks: "",
    fi_remarks: "",
    rcu_remarks: "",
    collection_remarks: "",
    stock_inspection_remarks: "",
  });
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    api.get("/notes/history", { params: { limit: 5 } }).then((r) => setHistory(r.data.items || []));
  }, []);

  // If borrower query param provided, pre-fill actual notes from borrower record
  useEffect(() => {
    if (borrower) {
      api.get(`/borrowers/${borrower}`).then((r) => {
        setNotes((n) => ({
          ...n,
          borrower_id: borrower,
          cam_remarks: r.data.remarks?.cam_remarks || "",
          fi_remarks: r.data.remarks?.fi_remarks || "",
          rcu_remarks: r.data.remarks?.rcu_remarks || "",
          collection_remarks: r.data.remarks?.collection_remarks || "",
          stock_inspection_remarks: r.data.remarks?.stock_inspection_remarks || "",
        }));
      });
    }
  }, [borrower]);

  const submit = async () => {
    setLoading(true);
    setAnalysis(null);
    try {
      const r = await api.post("/notes/analyze", notes);
      setAnalysis(r.data.analysis);
      toast.success("NLP analysis complete");
      api.get("/notes/history", { params: { limit: 5 } }).then((res) => setHistory(res.data.items || []));
    } catch (e) {
      toast.error(e.response?.data?.detail || "Failed to analyze");
    } finally {
      setLoading(false);
    }
  };

  const loadSample = () => setNotes((n) => ({ ...n, ...SAMPLE }));

  return (
    <div className="p-8 space-y-6" data-testid="notes-page">
      <Toaster position="top-right" theme="dark" />
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Unstructured Intelligence · LLM NLP
        </div>
        <h1 className="text-3xl font-heading font-light tracking-tight mt-1 flex items-center gap-3">
          Banker Notes Analyzer
          <Sparkles size={20} className="text-primary" />
        </h1>
        <p className="text-sm text-fg-muted mt-2 max-w-2xl">
          Paste CAM notes, field investigation reports, RCU observations, collection remarks or stock inspection
          notes. The engine extracts explainable stress and fraud signals and maps them to standardized reason
          codes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Input · Free-Text Banker Remarks" testId="notes-input">
          <div className="space-y-3">
            <TextInput
              label="Borrower ID (optional)"
              value={notes.borrower_id}
              onChange={(v) => setNotes({ ...notes, borrower_id: v })}
              testId="input-borrower-id"
            />
            <TextArea
              label="CAM Remarks"
              value={notes.cam_remarks}
              onChange={(v) => setNotes({ ...notes, cam_remarks: v })}
              testId="input-cam"
            />
            <TextArea
              label="Field Investigation"
              value={notes.fi_remarks}
              onChange={(v) => setNotes({ ...notes, fi_remarks: v })}
              testId="input-fi"
            />
            <TextArea
              label="RCU Observations"
              value={notes.rcu_remarks}
              onChange={(v) => setNotes({ ...notes, rcu_remarks: v })}
              testId="input-rcu"
            />
            <TextArea
              label="Collection Remarks"
              value={notes.collection_remarks}
              onChange={(v) => setNotes({ ...notes, collection_remarks: v })}
              testId="input-collection"
            />
            <TextArea
              label="Stock Inspection"
              value={notes.stock_inspection_remarks}
              onChange={(v) => setNotes({ ...notes, stock_inspection_remarks: v })}
              testId="input-stock"
            />
            <div className="flex gap-3 pt-2">
              <Btn onClick={submit} disabled={loading} testId="analyze-btn">
                {loading ? "Analyzing…" : "Analyze with LLM"}
              </Btn>
              <Btn variant="ghost" onClick={loadSample} testId="load-sample-btn">
                Load Sample Stress Notes
              </Btn>
            </div>
          </div>
        </Panel>

        <Panel title="Extracted Intelligence" testId="notes-output">
          {loading && <Loader label="Extracting signals" />}
          {!loading && !analysis && (
            <div className="text-xs uppercase tracking-widest2 text-fg-faint font-heading py-12 text-center">
              Awaiting input · Run analysis to see extracted signals
            </div>
          )}
          {analysis && <AnalysisView data={analysis} />}
        </Panel>
      </div>

      {history.length > 0 && (
        <Panel title="Recent Analyses" testId="notes-history">
          <div className="space-y-2">
            {history.map((h) => (
              <div key={h.id} className="border border-border bg-bg rounded-sm p-3 flex items-center justify-between">
                <div>
                  <div className="font-mono text-xs text-fg">{h.borrower_id || "manual"}</div>
                  <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
                    {new Date(h.created_at).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-fg">
                    Stress {h.analysis?.overall_stress_score ?? "—"}
                  </span>
                  <span className="text-xs text-fg-muted">
                    {h.analysis?.signals?.length || 0} signals
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

function TextInput({ label, value, onChange, testId }) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">{label}</label>
      <input
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-bg border border-border rounded-sm px-3 py-2 text-sm font-mono focus:border-primary focus:outline-none"
      />
    </div>
  );
}
function TextArea({ label, value, onChange, testId }) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">{label}</label>
      <textarea
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={2}
        className="mt-1 w-full bg-bg border border-border rounded-sm px-3 py-2 text-sm focus:border-primary focus:outline-none resize-y"
      />
    </div>
  );
}

function AnalysisView({ data }) {
  const stress = data.overall_stress_score ?? 0;
  const stressColor = stress >= 70 ? "#EF4444" : stress >= 40 ? "#F59E0B" : "#10B981";
  return (
    <div className="space-y-5" data-testid="analysis-view">
      <div className="grid grid-cols-2 gap-4">
        <div className="border border-border bg-bg rounded-sm p-3">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
            Overall Stress
          </div>
          <div className="font-mono text-3xl mt-1" style={{ color: stressColor }} data-testid="analysis-stress-score">
            {stress}
            <span className="text-fg-muted text-base ml-1">/100</span>
          </div>
        </div>
        <div className="border border-border bg-bg rounded-sm p-3">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
            Sentiment
          </div>
          <div className="text-xl font-heading font-medium mt-1 capitalize" data-testid="analysis-sentiment">
            {data.overall_sentiment || "—"}
          </div>
        </div>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading mb-2">
          Extracted Signals
        </div>
        <div className="space-y-2" data-testid="analysis-signals">
          {(data.signals || []).map((s, i) => (
            <div key={i} className="border border-border bg-bg rounded-sm p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="font-mono text-xs" style={{ color: SEV[s.severity] || "#94A3B8" }}>
                  {s.code}
                </div>
                <span className="text-[10px] uppercase tracking-widest2 font-heading" style={{ color: SEV[s.severity] }}>
                  {s.severity}
                </span>
              </div>
              <div className="text-sm mt-1 text-fg">{s.explanation}</div>
              {s.evidence && (
                <div className="text-xs text-fg-muted mt-1 italic">"{s.evidence}"</div>
              )}
            </div>
          ))}
          {(!data.signals || data.signals.length === 0) && (
            <div className="text-xs text-fg-faint uppercase tracking-widest2 font-heading">
              No adverse signals detected
            </div>
          )}
        </div>
      </div>

      {data.recommended_action && (
        <div className="border border-primary/40 bg-primary/5 rounded-sm p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle size={14} className="text-primary mt-0.5" />
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-primary font-heading">
                Recommended Action
              </div>
              <div className="text-sm text-fg mt-1" data-testid="analysis-action">
                {data.recommended_action}
              </div>
            </div>
          </div>
        </div>
      )}

      {data.summary && (
        <div>
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Summary</div>
          <div className="text-sm text-fg mt-1" data-testid="analysis-summary">{data.summary}</div>
        </div>
      )}
    </div>
  );
}
