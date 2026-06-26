import Plot from 'react-plotly.js'
import type { AnalyzeResponse } from '../types'
import { emotionColor, emotionEmoji } from '../emotionMeta'

interface Props {
  result: AnalyzeResponse
}

const MODALITIES = ['video', 'audio', 'text'] as const
const MODALITY_ICON: Record<string, string> = { video: '🎬', audio: '🎙', text: '📝' }

export default function ResultSection({ result }: Props) {
  const {
    predicted_emotion,
    confidence,
    probabilities,
    modality_predictions,
    contributions,
    transcript,
    frames,
    inference_ms,
  } = result

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`
  const accentColor = emotionColor(predicted_emotion)

  // Contribution bar chart data
  const contribKeys = [...MODALITIES] as string[]
  const contribVals = (MODALITIES as readonly string[]).map((k) => contributions[k as 'video' | 'audio' | 'text'])
  const contribColors = contribVals.map((v) => (v >= 0 ? '#00D9FF' : '#FF5C35'))

  // All-probabilities bar chart
  const probLabels = Object.keys(probabilities)
  const probVals = Object.values(probabilities)

  return (
    <section className="ts-card ts-result" aria-label="Analysis result">
      <h2 className="ts-section-title">
        <span className="ts-section-num">02</span> Result
        <span className="ts-infer-time">{inference_ms.toFixed(0)} ms inference</span>
      </h2>

      {/* Hero emotion */}
      <div className="ts-emotion-hero" style={{ '--accent': accentColor } as React.CSSProperties}>
        <span className="ts-emotion-emoji">{emotionEmoji(predicted_emotion)}</span>
        <div className="ts-emotion-label">
          <span className="ts-emotion-name" style={{ color: accentColor }}>
            {predicted_emotion}
          </span>
          <span className="ts-emotion-conf">{pct(confidence)} confidence</span>
        </div>
        <div className="ts-conf-bar-wrap" aria-label={`Confidence ${pct(confidence)}`}>
          <div
            className="ts-conf-bar-fill"
            style={{ width: pct(confidence), background: accentColor }}
          />
        </div>
      </div>

      {/* Fused probability distribution */}
      <div className="ts-chart-block">
        <h3 className="ts-chart-title">Fused probability distribution</h3>
        <Plot
          data={[
            {
              type: 'bar',
              x: probLabels,
              y: probVals,
              marker: {
                color: probLabels.map((l) => emotionColor(l)),
                opacity: 0.9,
              },
              hovertemplate: '%{x}: %{y:.1%}<extra></extra>',
            },
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Sora, sans-serif', color: '#A8B8CC', size: 12 },
            margin: { t: 8, r: 8, b: 40, l: 48 },
            xaxis: { gridcolor: '#1E2A3A', tickfont: { size: 11 } },
            yaxis: { gridcolor: '#1E2A3A', tickformat: '.0%', range: [0, 1] },
            height: 200,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Modality contributions */}
      <div className="ts-chart-block">
        <h3 className="ts-chart-title">Modality contribution (leave-one-out)</h3>
        <p className="ts-chart-sub">Drop in fused confidence when modality is removed. Negative = that modality hurts fusion.</p>
        <Plot
          data={[
            {
              type: 'bar',
              x: contribKeys,
              y: contribVals,
              marker: { color: contribColors, opacity: 0.9 },
              hovertemplate: '%{x}: %{y:+.3f}<extra></extra>',
            },
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Sora, sans-serif', color: '#A8B8CC', size: 12 },
            margin: { t: 8, r: 8, b: 40, l: 64 },
            xaxis: { gridcolor: '#1E2A3A' },
            yaxis: {
              gridcolor: '#1E2A3A',
              zeroline: true,
              zerolinecolor: '#2E3E52',
              tickformat: '+.3f',
            },
            height: 200,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Per-modality predictions */}
      <div className="ts-modality-grid">
        {MODALITIES.map((mod) => {
          const mp = modality_predictions[mod]
          const mc = emotionColor(mp.emotion)
          return (
            <div key={mod} className="ts-modality-card">
              <div className="ts-modality-header">
                <span className="ts-modality-icon">{MODALITY_ICON[mod]}</span>
                <span className="ts-modality-name">{mod}</span>
              </div>
              <div className="ts-modality-emotion" style={{ color: mc }}>
                {emotionEmoji(mp.emotion)} {mp.emotion}
              </div>
              <div className="ts-modality-conf-bar">
                <div className="ts-modality-conf-fill" style={{ width: pct(mp.confidence), background: mc }} />
              </div>
              <span className="ts-modality-conf-label">{pct(mp.confidence)}</span>
            </div>
          )
        })}
      </div>

      {/* Transcript */}
      {transcript && (
        <div className="ts-transcript-block">
          <h3 className="ts-chart-title">Transcript (Whisper)</h3>
          <blockquote className="ts-transcript">
            <span className="ts-transcript-quote">"</span>
            {transcript}
            <span className="ts-transcript-quote">"</span>
          </blockquote>
        </div>
      )}

      {/* Frames */}
      {frames.length > 0 && (
        <div className="ts-frames-block">
          <h3 className="ts-chart-title">Sampled frames ({frames.length})</h3>
          <div className="ts-frames-strip" role="list">
            {frames.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`Frame ${i + 1}`}
                className="ts-frame-thumb"
                loading="lazy"
                role="listitem"
              />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
