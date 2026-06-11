export default function SourceCard({ source, index }) {
  const sourceIcons = {
    google_drive: '📁',
    slack: '💬',
    email: '📧',
    notion: '📝',
    github: '🐙',
    upload: '📤',
  }

  const icon = sourceIcons[source.source] || '📄'

  return (
    <div className="source-card">
      <div className="source-card-header">
        <span className="source-icon">{icon}</span>
        <span className="source-index">[{index}]</span>
        <span className="source-title">{source.doc_title || 'Untitled Document'}</span>
        <span className="source-score">{(source.relevance_score * 100).toFixed(0)}%</span>
      </div>
      {source.parent_heading && (
        <div className="source-section">§ {source.parent_heading}</div>
      )}
      <div className="source-text">{source.chunk_text?.slice(0, 200)}…</div>
      <div className="source-meta">
        <span className="source-type">{source.source}</span>
      </div>
    </div>
  )
}
