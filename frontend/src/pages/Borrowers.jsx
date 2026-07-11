import React, { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, fmtInr, fmtPct, fmtNum } from "../lib/api";
import { Panel, GradeBadge, Btn, Loader } from "../components/ui";
import { Search } from "lucide-react";

const GRADES = ["Green", "Yellow", "Amber", "Red", "Black"];

export default function Borrowers() {
  const [sp, setSp] = useSearchParams();
  const [meta, setMeta] = useState(null);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    search: sp.get("search") || "",
    grade: sp.get("grade") || "",
    sector: sp.get("sector") || "",
    geography: sp.get("geography") || "",
    segment: sp.get("segment") || "",
    sort_by: "pd_12m",
    order: "desc",
  });
  const [page, setPage] = useState(0);
  const limit = 25;

  // Keep URL and filter state in sync when the URL changes (e.g., drill-down from Portfolio)
  useEffect(() => {
    setFilters((f) => ({
      ...f,
      search: sp.get("search") || "",
      grade: sp.get("grade") || "",
      sector: sp.get("sector") || "",
      geography: sp.get("geography") || "",
      segment: sp.get("segment") || "",
    }));
    setPage(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sp.toString()]);

  useEffect(() => {
    api.get("/metadata").then((r) => setMeta(r.data));
  }, []);

  useEffect(() => {
    setLoading(true);
    const params = { ...filters, limit, offset: page * limit };
    Object.keys(params).forEach((k) => params[k] === "" && delete params[k]);
    api
      .get("/borrowers", { params })
      .then((r) => {
        setRows(r.data.borrowers);
        setTotal(r.data.total);
      })
      .finally(() => setLoading(false));
  }, [filters, page]);

  const pageCount = useMemo(() => Math.ceil(total / limit), [total]);

  const updateFilter = (k, v) => {
    setFilters((f) => ({ ...f, [k]: v }));
    setPage(0);
    // Reflect to URL for shareability / back-button
    const next = new URLSearchParams(sp);
    if (v) next.set(k, v);
    else next.delete(k);
    setSp(next, { replace: true });
  };

  return (
    <div className="p-8 space-y-6" data-testid="borrowers-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
          Portfolio · Borrower Explorer
        </div>
        <h1 className="text-3xl font-heading font-light tracking-tight mt-1">Borrowers</h1>
      </div>

      {/* Filters */}
      <Panel testId="filter-bar">
        <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
          <div className="md:col-span-2 relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-faint"
            />
            <input
              data-testid="filter-search"
              type="text"
              placeholder="SEARCH ID OR NAME"
              value={filters.search}
              onChange={(e) => updateFilter("search", e.target.value)}
              className="w-full bg-bg border border-border rounded-sm pl-9 pr-3 py-2 text-xs uppercase tracking-widest2 placeholder:text-fg-faint focus:border-primary focus:outline-none font-heading"
            />
          </div>
          <FilterSelect
            testId="filter-grade"
            value={filters.grade}
            onChange={(v) => updateFilter("grade", v)}
            options={GRADES}
            placeholder="All Grades"
          />
          <FilterSelect
            testId="filter-sector"
            value={filters.sector}
            onChange={(v) => updateFilter("sector", v)}
            options={meta?.sectors || []}
            placeholder="All Sectors"
          />
          <FilterSelect
            testId="filter-geography"
            value={filters.geography}
            onChange={(v) => updateFilter("geography", v)}
            options={meta?.geographies || []}
            placeholder="All Geographies"
          />
          <FilterSelect
            testId="filter-segment"
            value={filters.segment}
            onChange={(v) => updateFilter("segment", v)}
            options={meta?.segments || []}
            placeholder="All Segments"
          />
        </div>
      </Panel>

      <Panel testId="borrowers-table">
        <div className="flex items-center justify-between mb-3">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
            {total.toLocaleString("en-IN")} accounts · page {page + 1} of {pageCount || 1}
          </div>
          <div className="flex gap-2">
            <Btn variant="ghost" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))} testId="pg-prev">
              Prev
            </Btn>
            <Btn variant="ghost" disabled={page + 1 >= pageCount} onClick={() => setPage((p) => p + 1)} testId="pg-next">
              Next
            </Btn>
          </div>
        </div>
        {loading ? (
          <Loader />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading border-b border-border">
                  <th className="text-left py-2 pr-3">Borrower</th>
                  <th className="text-left py-2 pr-3">Segment</th>
                  <th className="text-left py-2 pr-3">Sector</th>
                  <th className="text-left py-2 pr-3">Geo</th>
                  <th className="text-right py-2 pr-3">Outstanding</th>
                  <th className="text-right py-2 pr-3">DPD</th>
                  <th className="text-right py-2 pr-3">PD 12M</th>
                  <th className="text-right py-2 pr-3">Health</th>
                  <th className="text-left py-2 pl-3">Grade</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((b) => (
                  <tr
                    key={b.borrower_id}
                    className="border-b border-border hover:bg-bg-hover transition-colors"
                    data-testid={`borrower-row-${b.borrower_id}`}
                  >
                    <td className="py-3 pr-3">
                      <Link to={`/borrowers/${b.borrower_id}`} className="hover:text-primary">
                        <div className="font-mono text-xs">{b.borrower_id}</div>
                        <div className="text-fg-muted text-xs">{b.borrower_name}</div>
                      </Link>
                    </td>
                    <td className="py-3 pr-3 text-fg-muted text-xs">{b.segment}</td>
                    <td className="py-3 pr-3 text-fg-muted text-xs">{b.sector}</td>
                    <td className="py-3 pr-3 text-fg-muted text-xs">{b.geography}</td>
                    <td className="py-3 pr-3 text-right font-mono">{fmtInr(b.outstanding_amount)}</td>
                    <td className="py-3 pr-3 text-right font-mono">
                      {b.current_dpd ? Math.round(b.current_dpd) : 0}
                    </td>
                    <td className="py-3 pr-3 text-right font-mono">{fmtPct(b.pd_12m, 1)}</td>
                    <td className="py-3 pr-3 text-right font-mono">{fmtNum(b.health_score)}</td>
                    <td className="py-3 pl-3">
                      <GradeBadge grade={b.risk_grade} size="sm" />
                    </td>
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

function FilterSelect({ value, onChange, options, placeholder, testId }) {
  return (
    <select
      data-testid={testId}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-bg border border-border rounded-sm px-3 py-2 text-xs uppercase tracking-widest2 focus:border-primary focus:outline-none font-heading text-fg"
    >
      <option value="">{placeholder}</option>
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );
}
