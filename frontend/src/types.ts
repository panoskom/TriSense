export interface ModalityPrediction {
  emotion: string
  confidence: number
  probabilities: Record<string, number>
}

export interface AnalyzeResponse {
  predicted_emotion: string
  confidence: number
  probabilities: Record<string, number>
  modality_predictions: Record<'video' | 'audio' | 'text', ModalityPrediction>
  contributions: Record<'video' | 'audio' | 'text', number>
  transcript: string
  frames: string[]
  inference_ms: number
}

export interface ConfusionMatrix {
  labels: string[]
  matrix: number[][]
}

export interface ModelCard {
  checkpoint: string
  trained_on: string
  device: string
  num_test_clips: number
  test_accuracy: number
  macro_f1: number
  per_modality_accuracy: Record<string, number>
  confusion_matrix: ConfusionMatrix
  confusion_matrix_image: string | null
  random_baseline: number
  note: string
}

export interface GalleryItem {
  clip_id: string
  true_emotion: string
  predicted_emotion: string
  confidence: number
  correct: boolean
  modality_predictions: Record<'video' | 'audio' | 'text', ModalityPrediction>
  contributions: Record<'video' | 'audio' | 'text', number>
  transcript: string
  thumbnail: string | null
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  device: string
  checkpoint: string
}
