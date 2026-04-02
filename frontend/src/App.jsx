import { Routes, Route, NavLink } from 'react-router-dom'
import Prediction from './components/Prediction'
import Visualizations from './components/Visualizations'
import Retraining from './components/Retraining'
import Dashboard from './components/Dashboard'
import './App.css'

function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">🐾</span>
          <span className="brand-text">Cat vs Dog Classifier</span>
        </div>
        <div className="nav-links">
          <NavLink to="/" end>Prediction</NavLink>
          <NavLink to="/visualizations">Visualizations</NavLink>
          <NavLink to="/retraining">Retraining</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Prediction />} />
          <Route path="/visualizations" element={<Visualizations />} />
          <Route path="/retraining" element={<Retraining />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
