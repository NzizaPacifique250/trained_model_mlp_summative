import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL ? String(import.meta.env.VITE_API_URL).replace(/\/$/, '') : '/api'

export default function Prediction() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onDrop = useCallback((accepted) => {
    if (accepted.length === 0) return
    const f = accepted[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] },
    multiple: false,
  })

  const handlePredict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${API}/predict`, formData)
      setResult(res.data.result)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const getEmoji = (label) => {
    const l = label.toLowerCase()
    if (l === 'cat') return '🐱'
    if (l === 'dog') return '🐶'
    return '🔍'
  }

  const getResultStyle = (result) => {
    if (result.is_cat_dog) {
      return result.prediction === 'Cat'
        ? { background: 'linear-gradient(135deg, #ebf4ff, #c3dafe)', border: '2px solid #4a6cf7' }
        : { background: 'linear-gradient(135deg, #fff5f5, #fed7d7)', border: '2px solid #e53e3e' }
    }
    return { background: 'linear-gradient(135deg, #f0fff4, #c6f6d5)', border: '2px solid #38a169' }
  }

  return (
    <div>
      <div className="card">
        <h2>Image Classification</h2>
        <p style={{ color: '#718096', marginBottom: '1.5rem' }}>
          Upload any image — <strong>cat/dog images</strong> are classified by our custom-trained model,
          while other images use the general ImageNet classifier (1,000 categories).
        </p>

        {!preview ? (
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="icon">📷</div>
            <p><strong>Drag & drop an image here</strong></p>
            <p>or click to browse (JPG, PNG, WebP)</p>
          </div>
        ) : (
          <>
            <div className="image-preview">
              <img src={preview} alt="Upload preview" />
            </div>

            <div style={{ textAlign: 'center', marginTop: '1rem', display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? <><span className="spinner" /> Analyzing...</> : 'Classify Image'}
              </button>
              <button className="btn" onClick={reset} style={{ background: '#edf2f7' }}>
                Clear
              </button>
            </div>
          </>
        )}

        {error && (
          <div className="alert alert-error" style={{ marginTop: '1rem' }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{ marginTop: '1.5rem' }}>
            {/* Model badge */}
            <div style={{ textAlign: 'center', marginBottom: '0.75rem' }}>
              <span style={{
                display: 'inline-block',
                padding: '4px 14px',
                borderRadius: '20px',
                fontSize: '0.8rem',
                fontWeight: 600,
                background: result.is_cat_dog ? '#ebf4ff' : '#f0fff4',
                color: result.is_cat_dog ? '#2a4365' : '#22543d',
              }}>
                {result.is_cat_dog ? '🧠 Custom Fine-Tuned Model' : '🌐 ImageNet General Classifier'}
              </span>
            </div>

            {/* Top prediction */}
            <div style={{
              textAlign: 'center',
              padding: '1.5rem',
              borderRadius: '12px',
              ...getResultStyle(result),
            }}>
              <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>
                {getEmoji(result.prediction)} {result.prediction}
              </div>
              <div style={{ fontSize: '1.1rem', color: '#4a5568', marginTop: '0.25rem' }}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>

            {/* Top predictions bar chart */}
            {result.top_predictions && result.top_predictions.length > 0 && (
              <div style={{ marginTop: '1.5rem' }}>
                <h3 style={{ marginBottom: '0.75rem', fontSize: '1rem', color: '#4a5568' }}>
                  {result.is_cat_dog ? 'Classification Scores' : 'Top 5 Predictions'}
                </h3>
                {result.top_predictions.map((pred, i) => (
                  <div key={i} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '8px',
                  }}>
                    <span style={{
                      minWidth: '24px',
                      fontWeight: 700,
                      color: i === 0 ? '#4a6cf7' : '#a0aec0',
                      fontSize: '0.9rem',
                    }}>
                      #{i + 1}
                    </span>
                    <span style={{
                      minWidth: '180px',
                      fontWeight: i === 0 ? 600 : 400,
                      fontSize: '0.9rem',
                    }}>
                      {pred.label}
                    </span>
                    <div style={{
                      flex: 1,
                      height: '24px',
                      background: '#edf2f7',
                      borderRadius: '6px',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${Math.max(pred.confidence * 100, 1)}%`,
                        height: '100%',
                        background: i === 0
                          ? 'linear-gradient(90deg, #4a6cf7, #667eea)'
                          : '#cbd5e0',
                        borderRadius: '6px',
                        transition: 'width 0.5s ease',
                      }} />
                    </div>
                    <span style={{
                      minWidth: '55px',
                      textAlign: 'right',
                      fontWeight: 600,
                      fontSize: '0.85rem',
                      color: i === 0 ? '#4a6cf7' : '#718096',
                    }}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
