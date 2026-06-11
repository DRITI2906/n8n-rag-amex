import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import HistorySidebar from './components/HistorySidebar'

export default function App() {
  const [history, setHistory] = useState([])
  const [activeIndex, setActiveIndex] = useState(null)

  const addToHistory = (entry) => {
    setHistory(prev => [entry, ...prev].slice(0, 50))
    setActiveIndex(0)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-brand">
          <span className="brand-icon">🧠</span>
          <span className="brand-name">DocIntel</span>
          <span className="brand-tag">Enterprise RAG</span>
        </div>
        <nav className="header-nav">
          <a href="http://localhost:5678" target="_blank" rel="noreferrer" className="nav-link">
            n8n Workflows
          </a>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="nav-link">
            API Docs
          </a>
        </nav>
      </header>

      <div className="app-body">
        <HistorySidebar
          history={history}
          onSelect={setActiveIndex}
          activeIndex={activeIndex}
        />
        <main className="main-content">
          <ChatInterface onNewQuery={addToHistory} />
        </main>
      </div>
    </div>
  )
}
