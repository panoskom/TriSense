export const EMOTION_EMOJI: Record<string, string> = {
  neutral: '😐',
  calm: '😌',
  happy: '😄',
  sad: '😢',
  angry: '😠',
  fearful: '😨',
  disgust: '🤢',
  surprised: '😲',
}

export const EMOTION_COLOR: Record<string, string> = {
  neutral: '#8B9BB4',
  calm: '#4ECDC4',
  happy: '#FFD93D',
  sad: '#6C91BF',
  angry: '#FF5C35',
  fearful: '#B983FF',
  disgust: '#6BCB77',
  surprised: '#FF9F1C',
}

export function emotionEmoji(emotion: string): string {
  return EMOTION_EMOJI[emotion.toLowerCase()] ?? '🎭'
}

export function emotionColor(emotion: string): string {
  return EMOTION_COLOR[emotion.toLowerCase()] ?? '#00D9FF'
}
