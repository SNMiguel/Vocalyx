import ScoreBar from './ScoreBar'

const META = {
  accept:  { icon: '✓', label: 'Accepted',   color: 'var(--accept)',  bar: '#10b981' },
  reject:  { icon: '✕', label: 'Rejected',   color: 'var(--reject)',  bar: '#ef4444' },
  retry:   { icon: '↺', label: 'Try Again',  color: 'var(--retry)',   bar: '#f59e0b' },
  step_up: { icon: '🔐', label: 'Step Up Required', color: 'var(--step-up)', bar: '#f97316' },
}

export default function DecisionCard({ result }) {
  const meta = META[result.decision] ?? META.reject
  return (
    <div className={`decision-card ${result.decision}`}>
      <div className="decision-icon">{meta.icon}</div>
      <div className="decision-label" style={{ color: meta.color }}>{meta.label}</div>
      <div className="decision-explanation">{result.explanation}</div>
      <div className="decision-scores">
        <ScoreBar label="Speaker match" value={result.speaker_score} color={meta.bar} />
        <ScoreBar
          label="Spoof risk"
          value={result.spoof_score}
          color={result.spoof_score > 0.5 ? 'var(--reject)' : result.spoof_score > 0.3 ? 'var(--retry)' : 'var(--accept)'}
        />
      </div>
    </div>
  )
}
