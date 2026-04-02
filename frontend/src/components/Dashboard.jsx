import { useState, useEffect } from 'react'
import axios from 'axios'

const API = '/api'

function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h}h ${m}m ${s}s`
}

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(false)

  const fetchData = () => {
    axios.get(`${API}/health`)
      .then(res => { setHealth(res.data); setError(false) })
      .catch(() => setError(true))

    axios.get(`${API}/data/stats`)
      .then(res => setStats(res.data))
      .catch(() => {})
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  const totalImages = stats
    ? Object.values(stats).reduce(
        (sum, split) => sum + (split.cats || 0) + (split.dogs || 0),
        0
      )
    : '—'

  const trainTotal = stats
    ? (stats.train?.cats || 0) + (stats.train?.dogs || 0)
    : '—'

  const valTotal = stats
    ? (stats.validation?.cats || 0) + (stats.validation?.dogs || 0)
    : '—'

  return (
    <div>
      {/* Status */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div style={{ marginBottom: '0.5rem' }}>
            <span className={`status-badge ${error ? 'offline' : 'online'}`}>
              {error ? '● Offline' : '● Online'}
            </span>
          </div>
          <div className="metric-label">API Status</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">
            {health ? formatUptime(health.uptime_seconds) : '—'}
          </div>
          <div className="metric-label">Uptime</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{totalImages}</div>
          <div className="metric-label">Total Images</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">MobileNetV2</div>
          <div className="metric-label" style={{ fontSize: '0.8rem' }}>Architecture</div>
        </div>
      </div>

      {/* Dataset Breakdown */}
      <div className="card">
        <h2>Dataset Overview</h2>
        {stats ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={th}>Split</th>
                  <th style={th}>Cats</th>
                  <th style={th}>Dogs</th>
                  <th style={th}>Total</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats).map(([split, counts]) => (
                  <tr key={split} style={{ borderBottom: '1px solid #edf2f7' }}>
                    <td style={{ ...td, fontWeight: 600 }}>
                      {split.charAt(0).toUpperCase() + split.slice(1)}
                    </td>
                    <td style={td}>{counts.cats || 0}</td>
                    <td style={td}>{counts.dogs || 0}</td>
                    <td style={td}>{(counts.cats || 0) + (counts.dogs || 0)}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr style={{ borderTop: '2px solid #e2e8f0', fontWeight: 700 }}>
                  <td style={td}>Total</td>
                  <td style={td}>
                    {Object.values(stats).reduce((s, c) => s + (c.cats || 0), 0)}
                  </td>
                  <td style={td}>
                    {Object.values(stats).reduce((s, c) => s + (c.dogs || 0), 0)}
                  </td>
                  <td style={td}>{totalImages}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        ) : error ? (
          <div className="alert alert-error">
            Cannot reach the API. Make sure the backend is running on port 8000.
          </div>
        ) : (
          <p style={{ color: '#718096' }}>Loading...</p>
        )}
      </div>

      {/* API Endpoints */}
      <div className="card">
        <h2>API Endpoints</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={th}>Method</th>
                <th style={th}>Endpoint</th>
                <th style={th}>Description</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['GET', '/health', 'Health check + uptime'],
                ['POST', '/predict', 'Classify an uploaded image'],
                ['POST', '/upload_data', 'Bulk upload labelled images'],
                ['POST', '/retrain', 'Trigger background retraining'],
                ['GET', '/data/stats', 'Dataset class counts'],
                ['GET', '/model/history', 'Training history (acc/loss)'],
              ].map(([method, path, desc], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #edf2f7' }}>
                  <td style={td}>
                    <span style={{
                      background: method === 'GET' ? '#c6f6d5' : '#bee3f8',
                      color: method === 'GET' ? '#22543d' : '#2a4365',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontWeight: 600,
                      fontSize: '0.8rem',
                    }}>{method}</span>
                  </td>
                  <td style={{ ...td, fontFamily: 'monospace' }}>{path}</td>
                  <td style={td}>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

const th = { padding: '10px 12px', textAlign: 'left', color: '#4a5568' }
const td = { padding: '10px 12px' }
