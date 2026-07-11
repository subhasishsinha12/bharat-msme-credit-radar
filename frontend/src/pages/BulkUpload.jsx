import React, { useRef, useState } from "react";
import { api, fmtInr, fmtPct, fmtNum } from "../lib/api";
import { Panel, Btn, Loader, GradeBadge } from "../components/ui";
import { UploadCloud, Download } from "lucide-react";
import { toast, Toaster } from "sonner";

export default function BulkUpload() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState(null);

  const onFile = (e) => setFile(e.target.files?.[0] || null);

  const submit = async () => {
    if (!file) return toast.error("Choose a CSV file first");
    setLoading(true);
    setResults([]);
    setSummary(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await api.post("/borrowers/bulk-score", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const rows = r.data.results || [];
      setResults(rows);
      const totalExp = rows.reduce((s, x) => s + (x.outstanding_amount || 0), 0);
      const stress = rows.reduce((s, x) => s + (x.expected_stress_amount || 0), 0);
      const grades = rows.reduce((acc, x) => {
        acc[x.risk_grade] = (acc[x.risk_grade] || 0) + 1;
        return acc;
      }, {});
      setSummary({ total: rows.length, totalExp, stress, grades });
      toast.success(`Scored ${rows.length} borrowers`);
    } catch (e) {
      toast.error(e.response?.data?.detail || "Bulk score failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadTemplate = () => {
    const headers = [
      "borrower_id", "borrower_name", "segment", "sector", "geography", "loan_type",
      "sanctioned_limit", "outstanding_amount", "current_dpd", "emi_bounce_count_6m",
      "cc_utilization_avg_3m", "gst_turnover_growth_yoy", "gst_filing_delay_count_6m",
      "gstr1_vs_3b_mismatch_pct", "bank_credit_to_gst_sales_ratio", "bureau_score",
      "bureau_enquiry_count_3m", "buyer_concentration_top2_pct",
      "epfo_employee_count_change_6m", "cam_remarks",
    ];
    const sample = [
      "TEST001,Test Enterprise,Manufacturer,Textile,Surat,Cash Credit,5000000,3500000,15,2,88.5,-12.5,3,18.2,0.62,682,5,61,-18,cash flow stress visible",
    ];
    const csv = [headers.join(","), sample.join(",")].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "msme_scoring_template.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-8 space-y-6" data-testid="upload-page">
      <Toaster position="top-right" theme="dark" />
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Bulk Intake · CSV
        </div>
        <h1 className="text-3xl font-heading font-light tracking-tight mt-1">
          CSV Bulk Scoring
        </h1>
        <p className="text-sm text-fg-muted mt-2 max-w-2xl">
          Upload a CSV of MSME accounts. Missing fields are imputed from the training-time
          population median/mode; the more fields you provide, the higher the data quality
          score and the more precise the PD estimate.
        </p>
      </div>

      <Panel testId="upload-form">
        <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
          <label className="border border-dashed border-border-strong rounded-sm px-6 py-8 flex-1 cursor-pointer hover:border-primary hover:bg-bg-hover transition-colors" data-testid="upload-dropzone">
            <input ref={fileRef} type="file" accept=".csv" onChange={onFile} className="hidden" data-testid="file-input" />
            <div className="flex items-center gap-4">
              <UploadCloud size={28} className="text-fg-muted" strokeWidth={1.2} />
              <div>
                <div className="text-sm text-fg">
                  {file ? file.name : "Choose a CSV file to score"}
                </div>
                <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">
                  Must contain a borrower_id column
                </div>
              </div>
            </div>
          </label>
          <div className="flex flex-col gap-2">
            <Btn onClick={submit} disabled={!file || loading} testId="submit-upload">
              {loading ? "Scoring…" : "Score Portfolio"}
            </Btn>
            <Btn variant="ghost" onClick={downloadTemplate} testId="download-template">
              <Download size={14} className="mr-2" />
              Template CSV
            </Btn>
          </div>
        </div>
      </Panel>

      {loading && <Panel><Loader label="Scoring uploaded portfolio" /></Panel>}

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Panel><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Rows Scored</div><div className="font-mono text-3xl mt-1">{summary.total.toLocaleString("en-IN")}</div></div></Panel>
          <Panel><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Total Exposure</div><div className="font-mono text-3xl mt-1">{fmtInr(summary.totalExp)}</div></div></Panel>
          <Panel><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Expected Stress</div><div className="font-mono text-3xl mt-1 text-grade-red">{fmtInr(summary.stress)}</div></div></Panel>
          <Panel><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Grade Split</div><div className="flex flex-wrap gap-1 mt-1">{Object.entries(summary.grades).map(([g, c]) => (<div key={g} className="flex items-center gap-1"><GradeBadge grade={g} size="sm" /><span className="font-mono text-xs">{c}</span></div>))}</div></div></Panel>
        </div>
      )}

      {results.length > 0 && (
        <Panel title={`Scored Results · ${results.length} rows`} testId="upload-results">
          <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-bg-surface">
                <tr className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading border-b border-border">
                  <th className="text-left py-2 pr-3">Borrower</th>
                  <th className="text-left py-2 pr-3">Sector</th>
                  <th className="text-right py-2 pr-3">Outstanding</th>
                  <th className="text-right py-2 pr-3">PD 12M</th>
                  <th className="text-right py-2 pr-3">Health</th>
                  <th className="text-left py-2 pl-3">Grade</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.borrower_id} className="border-b border-border">
                    <td className="py-2 pr-3">
                      <div className="font-mono text-xs">{r.borrower_id}</div>
                      <div className="text-fg-muted text-xs">{r.borrower_name}</div>
                    </td>
                    <td className="py-2 pr-3 text-xs text-fg-muted">{r.sector || "—"}</td>
                    <td className="py-2 pr-3 text-right font-mono">{fmtInr(r.outstanding_amount)}</td>
                    <td className="py-2 pr-3 text-right font-mono">{fmtPct(r.pd_12m, 1)}</td>
                    <td className="py-2 pr-3 text-right font-mono">{fmtNum(r.health_score)}</td>
                    <td className="py-2 pl-3"><GradeBadge grade={r.risk_grade} size="sm" /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
