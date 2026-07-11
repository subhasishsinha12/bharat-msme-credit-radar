import React from "react";
import { NavLink } from "react-router-dom";
import { LayoutDashboard, Users, FileText, UploadCloud, Radar } from "lucide-react";
import { cn } from "../lib/api";

const NAV = [
  { to: "/", label: "Portfolio", icon: LayoutDashboard, end: true, testId: "nav-portfolio" },
  { to: "/borrowers", label: "Borrowers", icon: Users, testId: "nav-borrowers" },
  { to: "/notes", label: "Notes NLP", icon: FileText, testId: "nav-notes" },
  { to: "/upload", label: "CSV Score", icon: UploadCloud, testId: "nav-upload" },
];

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-bg text-fg flex">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-border bg-bg-surface flex flex-col">
        <div className="px-5 py-6 border-b border-border">
          <div className="flex items-center gap-2">
            <Radar size={20} className="text-primary" strokeWidth={1.5} />
            <div className="font-heading font-medium text-sm tracking-tight leading-tight">
              BHARAT MSME<br />
              <span className="text-fg-muted text-[10px] uppercase tracking-widest2">
                Credit Radar
              </span>
            </div>
          </div>
        </div>

        <nav className="flex-1 py-4">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              data-testid={item.testId}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-5 py-3 text-xs uppercase tracking-widest2 font-heading border-l-2 transition-colors duration-150",
                  isActive
                    ? "border-primary bg-bg-hover text-fg"
                    : "border-transparent text-fg-muted hover:text-fg hover:bg-bg-hover"
                )
              }
            >
              <item.icon size={14} strokeWidth={1.5} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-border">
          <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading">
            v1.0 · Prototype
          </div>
          <div className="text-[10px] text-fg-faint mt-1">
            IDBI Innovate 2026 · Track 04
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 border-b border-border bg-bg-surface flex items-center px-8 justify-between">
          <div className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 bg-grade-green animate-pulse rounded-full" />
            <span className="text-[10px] uppercase tracking-widest2 text-fg-muted font-heading">
              Live · Synthetic Portfolio
            </span>
          </div>
          <div className="text-[10px] uppercase tracking-widest2 text-fg-faint font-heading" data-testid="header-tagline">
            12-Month Predictive Default Intelligence
          </div>
        </header>
        <div className="flex-1 overflow-y-auto">{children}</div>
      </main>
    </div>
  );
}
