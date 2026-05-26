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
  v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, 1, true); v.setUint32(24, sr, true)
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true)
  v.setUint16(34, 16, true); ws(36, 'data')
  v.setUint32(40, dataLen, true)

  const samples = new Int16Array(buf, 44)
  for (let i = 0; i < pcm.length; i++) {
    samples[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32767))
  }
  return new Blob([buf], { type: 'audio/wav' })
}

/**
 * Returns a 0-100 live volume level while recording is active.
 * Pass the same `recording` boolean from useAudioRecorder so the analyser
 * only runs while the mic is open.
 */
export function useMicLevel(recording) {
  const [level, setLevel] = useState(0)
  const rafRef = useRef(null)
  const analyserRef = useRef(null)
  const streamRef = useRef(null)

  const attach = useCallback((stream) => {
    const actx = new AudioContext()
    const source = actx.createMediaStreamSource(stream)
    const analyser = actx.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    analyserRef.current = analyser
    streamRef.current = actx

    const data = new Uint8Array(analyser.frequencyBinCount)
    const tick = () => {
      analyser.getByteFrequencyData(data)
      const avg = data.reduce((s, v) => s + v, 0) / data.length
      setLevel(Math.round((avg / 255) * 100))
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()
  }, [])

  const detach = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.close()
    setLevel(0)
  }, [])

  return { level, attach, detach }
}

export function useAudioRecorder() {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [micError, setMicError] = useState('')
  const [micLevel, setMicLevel] = useState(0)

  const mrRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const t0Ref = useRef(null)
  const analyserRef = useRef(null)
  const actxRef = useRef(null)
  const rafRef = useRef(null)

  const start = useCallback(async () => {
    setMicError('')
    let stream

    // Use `ideal` (advisory) so drivers that can't honour the exact value
    // still give us a stream instead of rejecting the whole call.
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: { ideal: false },
          noiseSuppression: { ideal: false },
          autoGainControl: { ideal: false },
        },
        video: false,
      })
    } catch {
      // Fall back to default audio — some drivers reject any advanced constraints
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      } catch (err) {
        setMicError(
          err.name === 'NotAllowedError'
            ? 'Microphone access denied. Click the lock icon in your browser address bar and allow microphone access.'
            : `Could not access microphone: ${err.message}`
        )
        return
      }
    }

    const mr = new MediaRecorder(stream)
    mrRef.current = mr
    chunksRef.current = []

    mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.start(100)

    // Live mic level meter via AnalyserNode
    try {
      const actx = new AudioContext()
      const source = actx.createMediaStreamSource(stream)
      const analyser = actx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      actxRef.current = actx
      analyserRef.current = analyser
      const data = new Uint8Array(analyser.frequencyBinCount)
      const tick = () => {
        analyser.getByteFrequencyData(data)
        const avg = data.reduce((s, v) => s + v, 0) / data.length
        setMicLevel(Math.round((avg / 255) * 100))
        rafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch { /* level meter is non-critical */ }

    setRecording(true)
    setElapsed(0)
    t0Ref.current = Date.now()
    timerRef.current = setInterval(() => setElapsed((Date.now() - t0Ref.current) / 1000), 100)
  }, [])

  const stop = useCallback(() => new Promise(resolve => {
    const mr = mrRef.current
    if (!mr || mr.state === 'inactive') { resolve(null); return }

    mr.onstop = async () => {
      clearInterval(timerRef.current)
      cancelAnimationFrame(rafRef.current)
      actxRef.current?.close()
      setMicLevel(0)
      mr.stream.getTracks().forEach(t => t.stop())
      setRecording(false)

      if (chunksRef.current.length === 0) {
        setMicError('No audio was captured. Make sure your microphone is working and try again.')
        resolve(null)
        return
      }

      try {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' })
        const arrayBuf = await blob.arrayBuffer()
        const actx = new AudioContext()
        const audioBuffer = await actx.decodeAudioData(arrayBuf)
        await actx.close()
        const wavBlob = encodeWav(audioBuffer)
        resolve({ wavBlob, duration: audioBuffer.duration })
      } catch (e) {
        setMicError(`Failed to process recording — try a different browser or check your mic. (${e.message})`)
        resolve(null)
      }
    }

    mr.stop()
  }), [])

  return { recording, elapsed, micError, micLevel, start, stop }
}
