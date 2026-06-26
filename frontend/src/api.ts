import axios, { AxiosError } from 'axios'
import type { AnalyzeResponse, GalleryItem, HealthResponse, ModelCard } from './types'

const client = axios.create({ baseURL: '/api' })

function extractDetail(err: unknown): string {
  if (err instanceof AxiosError && err.response?.data?.detail) {
    const detail = err.response.data.detail
    return typeof detail === 'string' ? detail : JSON.stringify(detail)
  }
  if (err instanceof Error) return err.message
  return 'Unknown error'
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await client.get<HealthResponse>('/health')
    return res.data
  } catch (err) {
    throw new Error(extractDetail(err))
  }
}

export async function analyzeVideo(
  file: File,
  onProgress?: (pct: number) => void
): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('file', file)
  try {
    const res = await client.post<AnalyzeResponse>('/analyze', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      },
    })
    return res.data
  } catch (err) {
    throw new Error(extractDetail(err))
  }
}

export async function fetchModelCard(): Promise<ModelCard | null> {
  try {
    const res = await client.get<ModelCard>('/model-card')
    return res.data
  } catch (err) {
    if (err instanceof AxiosError && err.response?.status === 404) return null
    throw new Error(extractDetail(err))
  }
}

export async function fetchGallery(): Promise<GalleryItem[]> {
  try {
    const res = await client.get<GalleryItem[]>('/gallery')
    return res.data
  } catch (err) {
    throw new Error(extractDetail(err))
  }
}
