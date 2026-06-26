import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { fetchModelCard } from '../api'
import type { ModelCard } from '../types'
import { emotionColor } from '../emotionMeta'

function MetricTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="ts-metric-tile">
      <span className="ts-metric-value">{value}</span>
      <span className="ts-metric-label">{label}</span>
      {sub && <span className="ts-metric-sub">{sub}</span>}
    </div>
  )
}

export default function ModelCardSection() {
  const [card, setCard] = useState<ModelCard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notFound, setNotFound] = useState(false)

  useEffect(() => {
    fetchModelCard()
      .then((data) => {
        if (data === null) setNotFound(true)
        else setCard(data)
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <section className="ts-card" aria-label="Model card">
      <h2 className="ts-section-title">
        <span className="ts-section-num">04</span> Model Card
        <a
          href="http://localhost:5000"
          target="_blank"
          rel="noopener noreferrer"
          className="ts-mlflow-link"
          aria-label="Open MLflow training runs in new tab"
        >
          View training runs in MLflow ↗
        </a>
      </h2>

      {loading && (
        <div className="ts-placeholder">
          <div className="ts-spinner" aria-label="Loading model card" />
        </div>
      )}

      {error && <p className="ts-error-inline" role="alert">⚠ {error}</p>}

      {notFound && !error && (
        <div className="ts-placeholder ts-placeholder--model">
          <span className="ts-placeholder-icon">🧠</span>
          <p className="ts-placeholder-text">Model card available after training</p>
          <p className="ts-placeholder-sub">Train the model then re-run evaluation to generate metrics.</p>
        </div>
      )}

      {card && (
        <div className="ts-model-card-body">
          {/* Metric tiles */}
          <div className="ts-metrics-grid">
            <MetricTile
              label="Test Accuracy"
              value={`${(card.test_accuracy * 100).toFixed(1)}%`}
              sub={`vs ${(card.random_baseline * 100).toFixed(1)}% random`}
            />
            <MetricTile label="Macro F1" value={card.macro_f1.toFixed(3)} />
            <MetricTile label="Test Clips" value={card.num_test_clips.toString()} />
            <MetricTile label="Device" value={card.device} />
          </div>

          {/* Checkpoint / trained on */}
          <div className="ts-model-meta">
            <div className="ts-model-meta-row">
              <span className="ts-model-meta-key">Checkpoint</span>
              <code className="ts-model-meta-val">{card.checkpoint}</code>
            </div>
            <div className="ts-model-meta-row">
              <span className="ts-model-meta-key">Trained on</span>
              <code className="ts-model-meta-val">{card.trained_on}</code>
            </div>
          </div>

          {/* Per-modality accuracy */}
          <div className="ts-chart-block">
            <h3 className="ts-chart-title">Per-modality accuracy</h3>
            <div className="ts-modality-acc-list">
              {Object.entries(card.per_modality_accuracy).map(([mod, acc]) => (
                <div key={mod} className="ts-modality-acc-row">
                  <span className="ts-modality-acc-label">{mod}</span>
                  <div className="ts-modality-acc-bar-track">
                    <div
                      className="ts-modality-acc-bar-fill"
                      style={{ width: `${acc * 100}%`, background: '#00D9FF' }}
                    />
                  </div>
                  <span className="ts-modality-acc-val">{(acc * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Confusion matrix */}
          <div className="ts-chart-block">
            <h3 className="ts-chart-title">Confusion matrix</h3>
            {card.confusion_matrix_image ? (
              <img
                src={card.confusion_matrix_image}
                alt="Confusion matrix heatmap"
                className="ts-cm-image"
              />
            ) : (
              <Plot
                data={[
                  {
                    type: 'heatmap',
                    z: card.confusion_matrix.matrix,
                    x: card.confusion_matrix.labels,
                    y: card.confusion_matrix.labels,
                    colorscale: [
                      [0, '#080D1A'],
                      [0.5, '#004D66'],
                      [1, '#00D9FF'],
                    ],
                    showscale: true,
                    hovertemplate: 'True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
                    xgap: 2,
                    ygap: 2,
                  },
                ]}
                layout={{
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { family: 'Sora, sans-serif', color: '#A8B8CC', size: 11 },
                  margin: { t: 16, r: 16, b: 80, l: 80 },
                  xaxis: {
                    title: { text: 'Predicted', font: { size: 12 } },
                    tickfont: { size: 10 },
                    tickangle: -35,
                  },
                  yaxis: {
                    title: { text: 'True', font: { size: 12 } },
                    tickfont: { size: 10 },
                    autorange: 'reversed',
                  },
                  height: 380,
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler
              />
            )}
          </div>

          {/* Per-emotion colors reference for heatmap */}
          <div className="ts-emotion-legend">
            {card.confusion_matrix.labels.map((label) => (
              <span key={label} className="ts-legend-chip" style={{ borderColor: emotionColor(label) }}>
                <span className="ts-legend-dot" style={{ background: emotionColor(label) }} />
                {label}
              </span>
            ))}
          </div>

          {/* Note */}
          {card.note && (
            <div className="ts-model-note">
              <span className="ts-model-note-icon">ℹ</span>
              <p>{card.note}</p>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
