import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts'

const API = import.meta.env.VITE_API_URL ? String(import.meta.env.VITE_API_URL).replace(/\/$/, '') : '/api'

export default function Visualizations() {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState(null)
  const [statsErr, setStatsErr] = useState(null)
  const [histErr, setHistErr] = useState(null)

  useEffect(() => {
    axios.get(`${API}/data/stats`)
      .then(res => setStats(res.data))
      .catch(() => setStatsErr('Could not load dataset statistics. Is the API running?'))

    axios.get(`${API}/model/history`)
      .then(res => setHistory(res.data))
      .catch(() => setHistErr('No training history available. Train the model first.'))
  }, [])

  // Transform stats for Recharts
  const distData = stats
    ? Object.entries(stats).map(([split, counts]) => ({
        split: split.charAt(0).toUpperCase() + split.slice(1),
        Cats: counts.cats,
        Dogs: counts.dogs,
      }))
    : []

  // Transform history for Recharts
  const accData = history
    ? history.accuracy.map((v, i) => ({
        epoch: i + 1,
        'Train Acc': +(v * 100).toFixed(2),
        'Val Acc': +(history.val_accuracy[i] * 100).toFixed(2),
      }))
    : []

  const lossData = history
    ? history.loss.map((v, i) => ({
        epoch: i + 1,
        'Train Loss': +v.toFixed(4),
        'Val Loss': +history.val_loss[i].toFixed(4),
      }))
    : []

  return (
    <div>
      {/* ── Visualization 1: Class Distribution ── */}
      <div className="card">
        <h2>1. Class Distribution</h2>
        {statsErr ? (
          <div className="alert alert-warning">{statsErr}</div>
        ) : stats ? (
          <>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distData} barGap={8}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="split" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Cats" fill="#4a6cf7" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Dogs" fill="#e53e3e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="interpretation">
              <strong>Interpretation:</strong>
              The chart shows how cat and dog images are distributed across training and
              validation splits. A balanced distribution is critical — if one class
              dominated, the model could achieve high accuracy by always predicting the
              majority class. Our near-equal split ensures the model must learn genuine
              distinguishing features (fur texture, ear shape, snout) rather than
              exploiting class imbalance.
            </div>
          </>
        ) : (
          <p style={{ color: '#718096' }}>Loading statistics...</p>
        )}
      </div>

      {/* ── Visualization 2: Training Curves ── */}
      <div className="card">
        <h2>2. Training vs Validation Accuracy</h2>
        {histErr ? (
          <div className="alert alert-warning">{histErr}</div>
        ) : history ? (
          <>
            <div className="two-col">
              <div>
                <h3 style={{ textAlign: 'center' }}>Accuracy (%)</h3>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={accData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="Train Acc" stroke="#4a6cf7" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="Val Acc" stroke="#e53e3e" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3 style={{ textAlign: 'center' }}>Loss</h3>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="Train Loss" stroke="#4a6cf7" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="Val Loss" stroke="#e53e3e" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <div className="interpretation">
              <strong>Interpretation:</strong>
              The accuracy curve shows how quickly the model learns to distinguish cats
              from dogs. When training and validation lines stay close, the model
              generalises well. A widening gap signals overfitting. The loss curve should
              decrease steadily — sudden spikes may indicate the learning rate is too
              high. EarlyStopping and ReduceLROnPlateau callbacks help keep both curves
              healthy.
            </div>
          </>
        ) : (
          <p style={{ color: '#718096' }}>Loading training history...</p>
        )}
      </div>

      {/* ── Visualization 3: Per-epoch summary ── */}
      <div className="card">
        <h2>3. Per-Epoch Performance Summary</h2>
        {histErr ? (
          <div className="alert alert-warning">{histErr}</div>
        ) : history ? (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={th}>Epoch</th>
                    <th style={th}>Train Acc</th>
                    <th style={th}>Val Acc</th>
                    <th style={th}>Train Loss</th>
                    <th style={th}>Val Loss</th>
                    <th style={th}>Overfit Gap</th>
                  </tr>
                </thead>
                <tbody>
                  {history.accuracy.map((_, i) => {
                    const gap = (
                      (history.accuracy[i] - history.val_accuracy[i]) * 100
                    ).toFixed(1)
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #edf2f7' }}>
                        <td style={td}>{i + 1}</td>
                        <td style={td}>{(history.accuracy[i] * 100).toFixed(2)}%</td>
                        <td style={td}>{(history.val_accuracy[i] * 100).toFixed(2)}%</td>
                        <td style={td}>{history.loss[i].toFixed(4)}</td>
                        <td style={td}>{history.val_loss[i].toFixed(4)}</td>
                        <td style={{ ...td, color: gap > 5 ? '#e53e3e' : '#38a169' }}>
                          {gap > 0 ? '+' : ''}{gap}%
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="interpretation">
              <strong>Interpretation:</strong>
              The "Overfit Gap" column shows the difference between training and validation
              accuracy. A small positive gap ({'<'}5%) is normal. A large or growing gap
              means the model is memorising training data instead of learning generalisable
              patterns. Our callbacks (Dropout, EarlyStopping) keep this gap in check.
            </div>
          </>
        ) : (
          <p style={{ color: '#718096' }}>Loading...</p>
        )}
      </div>
    </div>
  )
}

const th = { padding: '10px 12px', textAlign: 'left', color: '#4a5568' }
const td = { padding: '10px 12px' }
