import { useState } from 'react'
import { submitFeedback } from '../api'

export default function FeedbackButtons({ queryId }) {
  const [voted, setVoted] = useState(null)
  const [loading, setLoading] = useState(false)

  const vote = async (rating) => {
    if (voted || loading || !queryId) return
    setLoading(true)
    try {
      await submitFeedback(queryId, rating)
      setVoted(rating)
    } catch (e) {
      console.error('Feedback failed:', e)
    } finally {
      setLoading(false)
    }
  }

  if (voted) {
    return (
      <div className="feedback-thanks">
        {voted === 1 ? '👍 Thanks for the feedback!' : '👎 Got it, we\'ll improve!'}
      </div>
    )
  }

  return (
    <div className="feedback-buttons">
      <span className="feedback-label">Was this helpful?</span>
      <button
        className="feedback-btn up"
        onClick={() => vote(1)}
        disabled={loading}
        title="Thumbs up"
      >
        👍
      </button>
      <button
        className="feedback-btn down"
        onClick={() => vote(-1)}
        disabled={loading}
        title="Thumbs down"
      >
        👎
      </button>
    </div>
  )
}
