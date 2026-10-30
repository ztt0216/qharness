import { Link, NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Runs from "./pages/Runs";
import RunDetail from "./pages/RunDetail";
import NewRun from "./pages/NewRun";
import Factors from "./pages/Factors";
import Models from "./pages/Models";
import Paper from "./pages/Paper";
import Reports from "./pages/Reports";

const NAV = [
  { to: "/", label: "总览", end: true },
  { to: "/reports", label: "研报来源" },
  { to: "/runs", label: "研究运行" },
  { to: "/factors", label: "因子库" },
  { to: "/models", label: "模型" },
  { to: "/paper", label: "模拟账户" },
];

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <Link to="/" className="wordmark"><b>因子研究台</b><span>沪深 300 成分股</span></Link>
          <nav className="nav" aria-label="主导航">
            {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>)}
          </nav>
          <Link to="/new" className="btn btn-primary" style={{ textDecoration: "none" }}>发起研究</Link>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/new" element={<NewRun />} />
          <Route path="/factors" element={<Factors />} />
          <Route path="/models" element={<Models />} />
          <Route path="/paper" element={<Paper />} />
          <Route path="*" element={<div className="page"><h1>没有这个页面</h1><p style={{ marginTop: 12 }}><Link to="/">回到总览</Link></p></div>} />
        </Routes>
      </main>
    </>
  );
}
