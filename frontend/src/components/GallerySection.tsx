import { useEffect, useState } from 'react'
import { fetchGallery } from '../api'
import type { GalleryItem } from '../types'
import { emotionColor, emotionEmoji } from '../emotionMeta'

const MODALITIES = ['video', 'audio', 'text'] as const

function MiniContribBars({ contributions }: { contributions: Record<string, number> }) {
  const max = Math.max(...Object.values(contributions).map(Math.abs), 0.001)
  return (
    <div className="ts-mini-contrib" aria-label="Modality contributions">
      {MODALITIES.map((mod) => {
        const val = contributions[mod] ?? 0
        const widthPct = Math.abs(val) / max * 100
        const isNeg = val < 0
        return (
          <div key={mod} className="ts-mini-contrib-row">
            <span className="ts-mini-contrib-label">{mod}</span>
            <div className="ts-mini-bar-track">
              <div
                className="ts-mini-bar-fill"
                style={{
                  width: `${widthPct}%`,
                  background: isNeg ? '#FF5C35' : '#00D9FF',
                }}
              />
            </div>
            <span className="ts-mini-contrib-val" style={{ color: isNeg ? '#FF5C35' : '#A8B8CC' }}>
              {val >= 0 ? '+' : ''}{val.toFixed(3)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function GalleryCard({ item }: { item: GalleryItem }) {
  const predColor = emotionColor(item.predicted_emotion)
  return (
    <article className={`ts-gallery-card ${item.correct ? 'ts-gallery-card--correct' : 'ts-gallery-card--wrong'}`}>
      <div className="ts-gallery-thumb-wrap">
        {item.thumbnail ? (
          <img
            src={item.thumbnail}
            alt={`Clip ${item.clip_id} thumbnail`}
            className="ts-gallery-thumb"
            loading="lazy"
          />
        ) : (
          <div className="ts-gallery-thumb-placeholder" aria-hidden="true">🎬</div>
        )}
        <span className={`ts-gallery-verdict ${item.correct ? 'ts-gallery-verdict--ok' : 'ts-gallery-verdict--fail'}`}
          aria-label={item.correct ? 'Correct prediction' : 'Incorrect prediction'}>
          {item.correct ? '✓' : '✗'}
        </span>
      </div>

      <div className="ts-gallery-body">
        <div className="ts-gallery-emotions">
          <div className="ts-gallery-emotion-row">
            <span className="ts-gallery-emotion-label">True</span>
            <span className="ts-gallery-emotion-val" style={{ color: emotionColor(item.true_emotion) }}>
              {emotionEmoji(item.true_emotion)} {item.true_emotion}
            </span>
          </div>
          <div className="ts-gallery-emotion-row">
            <span className="ts-gallery-emotion-label">Pred</span>
            <span className="ts-gallery-emotion-val" style={{ color: predColor }}>
              {emotionEmoji(item.predicted_emotion)} {item.predicted_emotion}
            </span>
          </div>
          <div className="ts-gallery-conf-row">
            <span className="ts-gallery-conf-num" style={{ color: predColor }}>
              {(item.confidence * 100).toFixed(1)}%
            </span>
            <div className="ts-gallery-conf-bar">
              <div className="ts-gallery-conf-fill" style={{ width: `${item.confidence * 100}%`, background: predColor }} />
            </div>
          </div>
        </div>

        <MiniContribBars contributions={item.contributions} />

        {item.transcript && (
          <p className="ts-gallery-transcript" title={item.transcript}>
            "{item.transcript.length > 80 ? item.transcript.slice(0, 80) + '…' : item.transcript}"
          </p>
        )}

        <span className="ts-gallery-clip-id">{item.clip_id}</span>
      </div>
    </article>
  )
}

export default function GallerySection() {
  const [items, setItems] = useState<GalleryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGallery()
      .then(setItems)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const correct = items.filter((i) => i.correct).length
  const acc = items.length > 0 ? ((correct / items.length) * 100).toFixed(1) : null

  return (
    <section className="ts-card" aria-label="Results gallery">
      <h2 className="ts-section-title">
        <span className="ts-section-num">03</span> Gallery
        {acc !== null && (
          <span className="ts-gallery-acc">
            {correct}/{items.length} correct — {acc}% accuracy
          </span>
        )}
      </h2>

      {loading && (
        <div className="ts-placeholder">
          <div className="ts-spinner" aria-label="Loading gallery" />
        </div>
      )}

      {error && <p className="ts-error-inline" role="alert">⚠ {error}</p>}

      {!loading && !error && items.length === 0 && (
        <div className="ts-placeholder ts-placeholder--gallery">
          <span className="ts-placeholder-icon">📊</span>
          <p className="ts-placeholder-text">Gallery will appear after training</p>
          <p className="ts-placeholder-sub">Run evaluation to populate test-set predictions here.</p>
        </div>
      )}

      {!loading && items.length > 0 && (
        <div className="ts-gallery-grid">
          {items.map((item) => (
            <GalleryCard key={item.clip_id} item={item} />
          ))}
        </div>
      )}
    </section>
  )
}
