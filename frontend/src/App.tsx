import { useState } from 'react'
import Header from './components/Header'
import UploadSection from './components/UploadSection'
import ResultSection from './components/ResultSection'
import GallerySection from './components/GallerySection'
import ModelCardSection from './components/ModelCardSection'
import type { AnalyzeResponse } from './types'

export default function App() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null)

  return (
    <div className="ts-app">
      <Header />
      <main className="ts-main">
        <UploadSection onResult={setResult} />
        {result && <ResultSection result={result} />}
        <GallerySection />
        <ModelCardSection />
      </main>
      <footer className="ts-footer">
        <span>TriSense · tri-modal emotion recognition</span>
        <span className="ts-footer-sep">·</span>
        <span>CLIP + wav2vec2 + Whisper, fused with LoRA</span>
      </footer>
    </div>
  )
}
