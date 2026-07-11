import axios from "axios";

const BASE_URL = process.env.REACT_APP_BACKEND_URL;

export const api = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: { "Content-Type": "application/json" },
});

export const fmtInr = (n) => {
  if (n === null || n === undefined || isNaN(n)) return "—";
  if (Math.abs(n) >= 1e7) return `\u20B9${(n / 1e7).toFixed(2)} Cr`;
  if (Math.abs(n) >= 1e5) return `\u20B9${(n / 1e5).toFixed(2)} L`;
  if (Math.abs(n) >= 1e3) return `\u20B9${(n / 1e3).toFixed(1)} K`;
  return `\u20B9${Number(n).toFixed(0)}`;
};

export const fmtPct = (n, digits = 1) => {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return `${(Number(n) * 100).toFixed(digits)}%`;
};

export const fmtNum = (n, digits = 1) => {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(digits);
};

export const GRADE_META = {
  Green: { color: "#10B981", label: "GREEN", bg: "rgba(16,185,129,0.08)" },
  Yellow: { color: "#FBBF24", label: "YELLOW", bg: "rgba(251,191,36,0.10)" },
  Amber: { color: "#F59E0B", label: "AMBER", bg: "rgba(245,158,11,0.10)" },
  Red: { color: "#EF4444", label: "RED", bg: "rgba(239,68,68,0.10)" },
  Black: { color: "#a1a1aa", label: "BLACK", bg: "rgba(63,63,70,0.30)" },
};

export const cn = (...classes) => classes.filter(Boolean).join(" ");
