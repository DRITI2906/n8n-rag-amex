export default function HistorySidebar({ history, onSelect, activeIndex }) {
  if (history.length === 0) {
    return (
      <aside className="sidebar">
        <div className="sidebar-header">Recent Queries</div>
        <div className="sidebar-empty">No queries yet. Ask something!</div>
      </aside>
    )
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">Recent Queries</div>
      <ul className="history-list">
        {history.map((item, i) => (
          <li
            key={i}
            className={`history-item ${i === activeIndex ? 'active' : ''}`}
            onClick={() => onSelect(i)}
          >
            <span className="history-q">{item.question}</span>
            <span className="history-time">
              {new Date(item.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </li>
        ))}
      </ul>
    </aside>
  )
}
