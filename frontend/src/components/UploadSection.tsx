import { useCallback, useRef, useState } from 'react'
import { analyzeVideo } from '../api'
import type { AnalyzeResponse } from '../types'

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']
const ALLOWED_EXTS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
const MAX_BYTES = 50 * 1024 * 1024

function validateFile(file: File): string | null {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase()
  if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXTS.includes(ext)) {
    return `Unsupported format "${ext}". Allowed: ${ALLOWED_EXTS.join(', ')}`
  }
  if (file.size > MAX_BYTES) {
    return `File is ${(file.size / 1024 / 1024).toFixed(1)} MB — max is 50 MB`
  }
  return null
}

interface Props {
  onResult: (result: AnalyzeResponse) => void
}

export default function UploadSection({ onResult }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [apiError, setApiError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = (f: File) => {
    const err = validateFile(f)
    if (err) {
      setValidationError(err)
      setFile(null)
    } else {
      setValidationError(null)
      setApiError(null)
      setFile(f)
    }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) accept(f)
  }, [])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) accept(f)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setApiError(null)
    setUploadPct(0)
    try {
      const result = await analyzeVideo(file, setUploadPct)
      onResult(result)
    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="ts-card" aria-label="Upload video clip">
      <h2 className="ts-section-title">
        <span className="ts-section-num">01</span> Upload Clip
      </h2>

      <div
        className={`ts-dropzone ${dragging ? 'ts-dropzone--active' : ''} ${file ? 'ts-dropzone--has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        aria-label="Drop video file or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_EXTS.join(',')}
          onChange={onInputChange}
          className="ts-file-input"
          aria-hidden="true"
          tabIndex={-1}
        />
        {file ? (
          <div className="ts-dropzone-file">
            <span className="ts-file-icon">🎬</span>
            <span className="ts-file-name">{file.name}</span>
            <span className="ts-file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
          </div>
        ) : (
          <div className="ts-dropzone-prompt">
            <span className="ts-drop-icon">⬆</span>
            <span className="ts-drop-label">Drop a video clip here</span>
            <span className="ts-drop-sub">or click to browse · {ALLOWED_EXTS.join(' ')} · max 50 MB</span>
          </div>
        )}
      </div>

      {validationError && (
        <p className="ts-error-inline" role="alert">{validationError}</p>
      )}

      {loading && (
        <div className="ts-loading-state" aria-live="polite">
          <div className="ts-spinner" aria-hidden="true" />
          <div className="ts-loading-text">
            <span>Analyzing…</span>
            <span className="ts-loading-sub">(this runs CLIP + wav2vec2 + Whisper on CPU, ~a few seconds)</span>
          </div>
          {uploadPct < 100 && (
            <div className="ts-progress-bar" role="progressbar" aria-valuenow={uploadPct} aria-valuemin={0} aria-valuemax={100}>
              <div className="ts-progress-fill" style={{ width: `${uploadPct}%` }} />
            </div>
          )}
        </div>
      )}

      {apiError && (
        <p className="ts-error-inline" role="alert">⚠ {apiError}</p>
      )}

      <button
        className="ts-btn ts-btn--primary"
        onClick={handleAnalyze}
        disabled={!file || loading}
        aria-busy={loading}
      >
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>
    </section>
  )
}
