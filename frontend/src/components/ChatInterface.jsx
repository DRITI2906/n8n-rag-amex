import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { queryStream, uploadDocument } from '../api'
import SourceCard from './SourceCard'
import FeedbackButtons from './FeedbackButtons'

export default function ChatInterface({ onNewQuery }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendQuestion = async (question) => {
    if (!question.trim() || loading) return
    setError(null)
    setLoading(true)

    const userMsg = { role: 'user', content: question, ts: Date.now() }
    const assistantMsg = {
      role: 'assistant',
      content: '',
      sources: [],
      queryId: null,
      retrieval_ms: null,
      generation_ms: null,
      ts: Date.now(),
      streaming: true,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    onNewQuery?.({ question, ts: Date.now() })

    try {
      await queryStream(
        question,
        {},
        (token) => {
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.content += token
            updated[updated.length - 1] = last
            return updated
          })
        },
        (meta) => {
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.sources = meta.sources || []
            last.queryId = meta.query_id
            last.retrieval_ms = meta.retrieval_time_ms
            updated[updated.length - 1] = last
            return updated
          })
        },
        (done) => {
          setMessages(prev => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.generation_ms = done.generation_time_ms
            last.streaming = false
            updated[updated.length - 1] = last
            return updated
          })
        }
      )
    } catch (e) {
      setError(e.message)
      setMessages(prev => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.content = `Error: ${e.message}`
        last.streaming = false
        updated[updated.length - 1] = last
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuestion(input)
    setInput('')
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const result = await uploadDocument(file)
      setMessages(prev => [
        ...prev,
        {
          role: 'system',
          content: `✅ Uploaded & indexed: **${result.title || file.name}** — ${result.already_exists ? 'already existed, skipped.' : 'ready to query!'}`,
          ts: Date.now(),
        },
      ])
    } catch (e) {
      setError(`Upload failed: ${e.message}`)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const copyAnswer = (text) => navigator.clipboard.writeText(text).catch(() => {})

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>Enterprise Document Intelligence</h2>
            <p>Ask any question about your indexed documents. Upload files using the 📎 button below.</p>
            <div className="example-questions">
              {['What is the remote work policy?', 'Summarise the Q3 financial report', 'How do I request annual leave?'].map(q => (
                <button key={q} className="example-q" onClick={() => sendQuestion(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            {msg.role === 'user' && (
              <div className="message-bubble user-bubble">{msg.content}</div>
            )}

            {msg.role === 'assistant' && (
              <div className="assistant-response">
                <div className="message-bubble assistant-bubble">
                  {msg.streaming && !msg.content && <span className="typing-indicator">●●●</span>}
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  {msg.streaming && msg.content && <span className="cursor">▌</span>}
                </div>

                {!msg.streaming && msg.content && (
                  <div className="response-actions">
                    <button className="action-btn" onClick={() => copyAnswer(msg.content)} title="Copy answer">
                      📋 Copy
                    </button>
                    {msg.retrieval_ms != null && (
                      <span className="latency">
                        Retrieval: {msg.retrieval_ms}ms
                        {msg.generation_ms != null && ` · Generation: ${msg.generation_ms}ms`}
                      </span>
                    )}
                    <FeedbackButtons queryId={msg.queryId} />
                  </div>
                )}

                {msg.sources?.length > 0 && (
                  <div className="sources-section">
                    <div className="sources-label">Sources ({msg.sources.length})</div>
                    <div className="sources-grid">
                      {(() => {
                        const maxScore = Math.max(...msg.sources.map(s => s.relevance_score))
                        return msg.sources.map((src, j) => (
                          <SourceCard
                            key={j}
                            source={src}
                            index={j + 1}
                            normalizedScore={maxScore > 0 ? src.relevance_score / maxScore : 0}
                          />
                        ))
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )}

            {msg.role === 'system' && (
              <div className="system-message">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-banner">{error} <button onClick={() => setError(null)}>✕</button></div>}

      <form className="input-bar" onSubmit={handleSubmit}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.pptx,.txt,.md"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />
        <button
          type="button"
          className="upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title="Upload document"
        >
          {uploading ? '⏳' : '📎'}
        </button>
        <input
          className="question-input"
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={loading}
        />
        <button type="submit" className="send-btn" disabled={loading || !input.trim()}>
          {loading ? '⏳' : '→'}
        </button>
      </form>
    </div>
  )
}
