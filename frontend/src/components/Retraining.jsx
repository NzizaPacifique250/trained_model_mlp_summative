import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL ? String(import.meta.env.VITE_API_URL).replace(/\/$/, '') : '/api'

export default function Retraining() {
  const [label, setLabel] = useState('cats')
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [retraining, setRetraining] = useState(false)
  const [message, setMessage] = useState(null)

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    multiple: true,
    onDrop: (accepted) => {
      setFiles(prev => [...prev, ...accepted])
      setMessage(null)
    },
  })

  const handleUpload = async () => {
    if (files.length === 0) return
    setUploading(true)
    setMessage(null)

    const formData = new FormData()
    formData.append('label', label)
    files.forEach(f => formData.append('files', f))

    try {
      const res = await axios.post(`${API}/upload_data`, formData)
      setMessage({ type: 'success', text: res.data.message })
      setFiles([])
    } catch (err) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Upload failed.',
      })
    } finally {
      setUploading(false)
    }
  }

  const handleRetrain = async () => {
    setRetraining(true)
    setMessage(null)
    try {
      const res = await axios.post(`${API}/retrain`)
      setMessage({ type: 'success', text: res.data.message })
    } catch (err) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Retraining trigger failed.',
      })
    } finally {
      setRetraining(false)
    }
  }

  return (
    <div>
      {/* Upload Section */}
      <div className="card">
        <h2>Upload Training Data</h2>
        <p style={{ color: '#718096', marginBottom: '1rem' }}>
          Add new labelled images to the training set, then trigger retraining.
        </p>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ fontWeight: 600, marginRight: '0.75rem' }}>Label:</label>
          <select
            className="form-select"
            value={label}
            onChange={e => setLabel(e.target.value)}
          >
            <option value="cats">Cats</option>
            <option value="dogs">Dogs</option>
          </select>
        </div>

        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="icon">📁</div>
          <p><strong>Drag & drop images here</strong></p>
          <p>or click to browse (multiple files supported)</p>
        </div>

        {files.length > 0 && (
          <>
            <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
              {files.length} file{files.length > 1 ? 's' : ''} selected:
            </p>
            <ul className="file-list" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {files.map((f, i) => (
                <li key={i}>📎 {f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
              ))}
            </ul>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                className="btn btn-primary"
                onClick={handleUpload}
                disabled={uploading}
              >
                {uploading ? <><span className="spinner" /> Uploading...</> : `Upload ${files.length} file${files.length > 1 ? 's' : ''}`}
              </button>
              <button
                className="btn"
                style={{ background: '#edf2f7' }}
                onClick={() => { setFiles([]); setMessage(null) }}
              >
                Clear
              </button>
            </div>
          </>
        )}
      </div>

      {/* Retrain Section */}
      <div className="card">
        <h2>Trigger Model Retraining</h2>
        <p style={{ color: '#718096', marginBottom: '1.5rem' }}>
          After uploading new data, click below to retrain the model in the background.
          The API will continue serving predictions with the current model until
          retraining completes.
        </p>
        <button
          className="btn btn-success"
          onClick={handleRetrain}
          disabled={retraining}
          style={{ fontSize: '1.05rem', padding: '12px 32px' }}
        >
          {retraining ? <><span className="spinner" /> Starting...</> : 'Trigger Retraining'}
        </button>
      </div>

      {message && (
        <div className={`alert alert-${message.type}`}>
          {message.text}
        </div>
      )}
    </div>
  )
}
