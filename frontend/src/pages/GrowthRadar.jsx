import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtInr, fmtPct, fmtNum } from "../lib/api";
import { Panel, Loader, Btn } from "../components/ui";
import { TrendingUp, Sparkles } from "lucide-react";

const BAND_COLORS = {
  Priority: "#10B981",
  Hot: "#22D3EE",
  Emerging: "#818CF8",
  Passive: "#94A3B8",
  Dormant: "#475569",
};

const BANDS = ["Priority", "Hot", "Emerging", "Passive", "Dormant"];

export default function GrowthRadar() {
  const [summary, setSummary] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ band: "", sector: "", min_score: 55 });
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    api.get("/growth/summary").then((r) => setSummary(r.data));
    api.get("/metadata").then((r) => setMeta(r.data));
  }, []);

  useEffect(() => {
    setLoading(true);
    const params = { ...filters, limit: 50 };
    Object.keys(params).forEach((k) => params[k] === "" && delete params[k]);
    api
      .get("/growth/candidates", { params })
      .then((r) => setCandidates(r.data.candidates))
      .finally(() => setLoading(false));
  }, [filters]);

  return (
    <div className="p-8 space-y-6" data-testid="growth-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Revenue Intelligence · Growth Propensity Engine
        </div>
        <h1 className="text-3xl font-heading font-light tracking-tight mt-1 flex items-center gap-3">
          Growth Radar
          <TrendingUp size={20} className="text-grade-green" />
        </h1>
        <p className="text-sm text-fg-muted mt-2 max-w-3xl">
          Healthy MSMEs likely to need enhanced working capital or a new term loan within
          6–12 months. Same data rails as the default engine — inverted lens: cash-flow
          strength, GST growth momentum, utilization headroom, bureau posture, EPFO
          expansion. Emits a suggested product, indicative quantum, and RM outreach window.
        </p>
      </div>

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Panel testId="g-total"><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Total Candidates</div><div className="font-mono text-4xl mt-2 font-medium">{summary.total_candidates.toLocaleString("en-IN")}</div></div></Panel>
          <Panel testId="g-hot"><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Priority + Hot</div><div className="font-mono text-4xl mt-2 font-medium text-grade-green">{summary.hot_candidates.toLocaleString("en-IN")}</div></div></Panel>
          <Panel testId="g-pipeline"><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Revenue Pipeline</div><div className="font-mono text-4xl mt-2 font-medium text-primary">{fmtInr(summary.revenue_pipeline)}</div><div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading mt-1">Indicative quantum · candidates ≥55</div></div></Panel>
          <Panel testId="g-avg"><div><div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Avg Growth Score</div><div className="font-mono text-4xl mt-2 font-medium">{fmtNum(summary.average_growth_score)}</div></div></Panel>
        </div>
      )}

      {summary && (
        <Panel title="Propensity Band Distribution" testId="band-dist">
          <div className="grid grid-cols-5 gap-3">
            {BANDS.map((b) => {
              const c = summary.band_counts[b] || 0;
              return (
                <button
                  key={b}
                  onClick={() => setFilters((f) => ({ ...f, band: f.band === b ? "" : b }))}
                  className={`text-left border rounded-sm p-3 transition-colors ${filters.band === b ? "border-primary bg-primary/10" : "border-border bg-bg hover:bg-bg-hover"}`}
                  data-testid={`band-cell-${b}`}
                >
                  <div className="text-[10px] uppercase tracking-widest2 font-heading" style={{ color: BAND_COLORS[b] }}>{b}</div>
                  <div className="font-mono text-2xl mt-1">{c}</div>
                </button>
              );
            })}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div>
              <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading mb-2">Suggested Products</div>
              <div className="space-y-1">
                {Object.entries(summary.top_products || {}).map(([p, c]) => (
                  <div key={p} className="flex justify-between text-sm border-b border-border py-1">
                    <span className="text-fg">{p}</span>
                    <span className="font-mono text-fg-muted">{c}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Panel>
      )}

      <Panel title="Filters" testId="growth-filters">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <label className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">Min Score</label>
            <input
              type="range"
              min="0"
              max="100"
              value={filters.min_score}
              onChange={(e) => setFilters({ ...filters, min_score: Number(e.target.value) })}
              className="w-full mt-2 accent-primary"
              data-testid="min-score-slider"
            />
            <div className="font-mono text-xs text-fg mt-1">≥ {filters.min_score}</div>
          </div>
          <FilterSel value={filters.band} options={BANDS} placeholder="All Bands" onChange={(v) => setFilters({ ...filters, band: v })} testId="filter-band" />
          <FilterSel value={filters.sector} options={meta?.sectors || []} placeholder="All Sectors" onChange={(v) => setFilters({ ...filters, sector: v })} testId="filter-sector-g" />
          <FilterSel value={filters.geography} options={meta?.geographies || []} placeholder="All Geographies" onChange={(v) => setFilters({ ...filters, geography: v })} testId="filter-geo-g" />
        </div>
      </Panel>

      <Panel title={`Candidates · ${candidates.length} shown`} testId="growth-table">
        {loading ? (
          <Loader />
        ) : candidates.length === 0 ? (
          <div className="text-xs uppercase tracking-widest2 text-fg-faint font-heading py-12 text-center">
            No candidates match current filters
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading border-b border-border">
                  <th className="text-left py-2 pr-3">Borrower</th>
                  <th className="text-left py-2 pr-3">Sector · Geo</th>
                  <th className="text-right py-2 pr-3">Turnover ↑YoY</th>
                  <th className="text-right py-2 pr-3">CC Util</th>
                  <th className="text-right py-2 pr-3">Bureau</th>
                  <th className="text-right py-2 pr-3">Growth Score</th>
                  <th className="text-left py-2 pr-3">Suggested Product</th>
                  <th className="text-right py-2 pr-3">Quantum</th>
                  <th className="text-left py-2 pl-3">Window</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => (
                  <tr key={c.borrower_id} className="border-b border-border hover:bg-bg-hover" data-testid={`growth-row-${c.borrower_id}`}>
                    <td className="py-3 pr-3">
                      <Link to={`/borrowers/${c.borrower_id}`} className="hover:text-primary">
                        <div className="font-mono text-xs">{c.borrower_id}</div>
                        <div className="text-fg-muted text-xs">{c.borrower_name}</div>
                      </Link>
                    </td>
                    <td className="py-3 pr-3 text-xs text-fg-muted">
                      <div>{c.sector}</div>
                      <div className="text-fg-faint">{c.geography}</div>
                    </td>
                    <td className="py-3 pr-3 text-right font-mono text-grade-green">
                      {c.gst_turnover_growth_yoy > 0 ? "+" : ""}{fmtNum(c.gst_turnover_growth_yoy, 1)}%
                    </td>
                    <td className="py-3 pr-3 text-right font-mono">{fmtNum(c.cc_utilization_avg_3m, 0)}%</td>
                    <td className="py-3 pr-3 text-right font-mono">{fmtNum(c.bureau_score, 0)}</td>
                    <td className="py-3 pr-3 text-right font-mono text-lg" style={{ color: BAND_COLORS[c.growth_band] }}>{fmtNum(c.growth_score)}</td>
                    <td className="py-3 pr-3">
                      <div className="text-fg text-xs">{c.suggested_product}</div>
                      <div className="text-[10px] text-fg-faint uppercase tracking-widest2 font-heading" style={{ color: BAND_COLORS[c.growth_band] }}>{c.growth_band}</div>
                    </td>
                    <td className="py-3 pr-3 text-right font-mono text-primary">{fmtInr(c.indicative_quantum)}</td>
                    <td className="py-3 pl-3 font-heading text-[10px] uppercase tracking-widest2 text-fg-muted">{c.outreach_window}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}

function FilterSel({ value, onChange, options, placeholder, testId }) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">{placeholder}</label>
      <select
        data-testid={testId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full bg-bg border border-border rounded-sm px-3 py-2 text-xs uppercase tracking-widest2 focus:border-primary focus:outline-none font-heading text-fg"
      >
        <option value="">{placeholder}</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </div>
  );
}
