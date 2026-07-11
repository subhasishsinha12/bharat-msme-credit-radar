/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#050505",
          surface: "#0A0A0A",
          panel: "#121212",
          hover: "#161616",
        },
        border: {
          DEFAULT: "#27272A",
          strong: "#3F3F46",
        },
        fg: {
          DEFAULT: "#F8FAFC",
          muted: "#94A3B8",
          faint: "#475569",
        },
        primary: "#2563EB",
        grade: {
          green: "#10B981",
          yellow: "#FBBF24",
          amber: "#F59E0B",
          red: "#EF4444",
          black: "#3F3F46",
        },
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        heading: ["'Chivo'", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      letterSpacing: {
        widest2: "0.2em",
      },
      borderRadius: {
        sm: "2px",
      },
    },
  },
  plugins: [],
};
