import { useState, useRef, useCallback } from 'react'

function encodeWav(audioBuffer) {
  const sr = audioBuffer.sampleRate
  // Mix to mono
  let pcm = audioBuffer.getChannelData(0)
  if (audioBuffer.numberOfChannels > 1) {
    pcm = new Float32Array(audioBuffer.length)
    for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
      const ch = audioBuffer.getChannelData(c)
      for (let i = 0; i < pcm.length; i++) pcm[i] += ch[i]
    }
    for (let i = 0; i < pcm.length; i++) pcm[i] /= audioBuffer.numberOfChannels
  }

  const dataLen = pcm.length * 2
  const buf = new ArrayBuffer(44 + dataLen)
  const v = new DataView(buf)
  const ws = (off, s) => [...s].forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)))

  ws(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true)
  ws(8, 'WAVE'); ws(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true)   // PCM
  v.setUint16(22, 1, true); v.setUint32(24, sr, true)   // mono, sample rate
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true) // byte rate, block align
  v.setUint16(34, 16, true); ws(36, 'data')              // bits per sample
  v.setUint32(40, dataLen, true)

  const samples = new Int16Array(buf, 44)
  for (let i = 0; i < pcm.length; i++) {
    samples[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32767))
  }
  return new Blob([buf], { type: 'audio/wav' })
}

export function useAudioRecorder() {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [micError, setMicError] = useState('')

  const mrRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const t0Ref = useRef(null)

  const start = useCallback(async () => {
    setMicError('')
    try {
      // Disable browser's built-in DSP so our backend denoiser gets raw audio
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      })
      const mr = new MediaRecorder(stream)
      mrRef.current = mr
      chunksRef.current = []

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.start(100)

      setRecording(true)
      setElapsed(0)
      t0Ref.current = Date.now()
      timerRef.current = setInterval(() => setElapsed((Date.now() - t0Ref.current) / 1000), 100)
    } catch (err) {
      setMicError(
        err.name === 'NotAllowedError'
          ? 'Microphone access denied. Please allow microphone access and try again.'
          : `Microphone error: ${err.message}`
      )
    }
  }, [])

  const stop = useCallback(() => new Promise(resolve => {
    const mr = mrRef.current
    if (!mr || mr.state === 'inactive') { resolve(null); return }

    mr.onstop = async () => {
      clearInterval(timerRef.current)
      mr.stream.getTracks().forEach(t => t.stop())
      setRecording(false)
      try {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' })
        const arrayBuf = await blob.arrayBuffer()
        const actx = new AudioContext()
        const audioBuffer = await actx.decodeAudioData(arrayBuf)
        await actx.close()
        const wavBlob = encodeWav(audioBuffer)
        resolve({ wavBlob, duration: audioBuffer.duration })
      } catch {
        resolve(null)
      }
    }
    mr.stop()
  }), [])

  return { recording, elapsed, micError, start, stop }
}
