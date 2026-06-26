import { useEffect, useState } from 'react'
import { fetchHealth } from '../api'
import type { HealthResponse } from '../types'

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setError(true))
  }, [])

  return (
    <header className="ts-header">
      <div className="ts-header-brand">
        <div className="ts-logo">
          <span className="ts-logo-tri">Tri</span>
          <span className="ts-logo-sense">Sense</span>
        </div>
        <p className="ts-tagline">
          Tri-modal emotion recognition — video + audio + text, fused with LoRA
        </p>
      </div>
      <div className="ts-health-badge">
        {error ? (
          <span className="badge badge--error">⚠ Backend offline</span>
        ) : health ? (
          <span className={`badge ${health.model_loaded ? 'badge--ok' : 'badge--warn'}`}>
            <span className="badge-dot" />
            {health.model_loaded ? 'Model ready' : 'Loading model'}
            <span className="badge-sep">·</span>
            <span className="badge-device">{health.device}</span>
          </span>
        ) : (
          <span className="badge badge--idle">
            <span className="badge-dot badge-dot--pulse" />
            Connecting…
          </span>
        )}
      </div>
    </header>
  )
}
