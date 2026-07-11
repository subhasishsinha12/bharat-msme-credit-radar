import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Portfolio from "./pages/Portfolio";
import Borrowers from "./pages/Borrowers";
import BorrowerDetail from "./pages/BorrowerDetail";
import NotesAnalyzer from "./pages/NotesAnalyzer";
import BulkUpload from "./pages/BulkUpload";
import GrowthRadar from "./pages/GrowthRadar";
import OfficerMemo from "./pages/OfficerMemo";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Memo route is standalone (no sidebar, print-friendly) */}
        <Route path="/borrowers/:id/memo" element={<OfficerMemo />} />
        <Route
          path="*"
          element={
            <Layout>
              <Routes>
                <Route path="/" element={<Portfolio />} />
                <Route path="/borrowers" element={<Borrowers />} />
                <Route path="/borrowers/:id" element={<BorrowerDetail />} />
                <Route path="/growth" element={<GrowthRadar />} />
                <Route path="/notes" element={<NotesAnalyzer />} />
                <Route path="/upload" element={<BulkUpload />} />
              </Routes>
            </Layout>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
