import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Portfolio from "./pages/Portfolio";
import Borrowers from "./pages/Borrowers";
import BorrowerDetail from "./pages/BorrowerDetail";
import NotesAnalyzer from "./pages/NotesAnalyzer";
import BulkUpload from "./pages/BulkUpload";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Portfolio />} />
          <Route path="/borrowers" element={<Borrowers />} />
          <Route path="/borrowers/:id" element={<BorrowerDetail />} />
          <Route path="/notes" element={<NotesAnalyzer />} />
          <Route path="/upload" element={<BulkUpload />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
