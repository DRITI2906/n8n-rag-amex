const BASE = import.meta.env.VITE_API_URL || '/api'

export async function queryStream(question, options = {}, onToken, onMetadata, onDone) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      question,
      top_k: options.topK ?? 5,
      stream: true,
      source: options.source ?? null,
      provider: options.provider ?? null,
    }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()

    for (const line of lines) {
      if (line.startsWith('event: token')) continue
      if (line.startsWith('event: metadata')) continue
      if (line.startsWith('event: done')) continue
      if (line.startsWith('data: ')) {
        const raw = line.slice(6)
        try {
          const parsed = JSON.parse(raw)
          if (parsed.sources !== undefined) onMetadata?.(parsed)
          else if (parsed.generation_time_ms !== undefined) onDone?.(parsed)
        } catch {
          onToken?.(raw)
        }
      }
    }
  }
}

export async function querySync(question, options = {}) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      top_k: options.topK ?? 5,
      stream: false,
      source: options.source ?? null,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function submitFeedback(queryId, rating, userId = null, comment = null) {
  const res = await fetch(`${BASE}/query/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_id: queryId, rating, user_id: userId, comment }),
  })
  return res.json()
}

export async function uploadDocument(file, source = 'upload') {
  const form = new FormData()
  form.append('file', file)
  form.append('source', source)
  const res = await fetch(`${BASE}/parse_document`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`)
  const parsed = await res.json()

  // Chain: chunk → embed
  await fetch(`${BASE}/chunk_document`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: parsed.doc_id }),
  })
  await fetch(`${BASE}/embed_chunks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: parsed.doc_id }),
  })
  return parsed
}
