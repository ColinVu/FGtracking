import React, { useState, useEffect, useRef } from 'react'

function PixelScene({ scale = 1.5 }) {
  // compact canvas to fit between the title and the lines below
  const s = 8 * scale
  const W = 28  // grid width (smaller than before)
  const H = 16  // grid height

  const pixel = (x, y, w = 1, h = 1, color = '#000') => (
    <div
      key={`${x}-${y}-${w}-${h}-${color}`}
      style={{
        position: 'absolute',
        left: `${x * s}px`,
        top: `${y * s}px`,
        width: `${w * s}px`,
        height: `${h * s}px`,
        background: color,
        imageRendering: 'pixelated'
      }}
    />
  )

  const blocks = []
  // palette
  const post = '#FFD700'
  const pad1 = '#b62f4a'
  const pad2 = '#e24c66'
  const base = '#3b0f0f'

  // center the goalpost in the smaller canvas
  const ax = Math.round((W - 8) / 2) 
  const groundY = 12               // visual baseline (no actual ground drawn)

  // base shoe
  blocks.push(pixel(ax + 3, groundY + 1, 2, 1, base))

  // pole
  blocks.push(pixel(ax + 3, groundY - 4, 2, 5, post))

  // padding bands
  blocks.push(pixel(ax + 3, groundY - 3, 2, 1, pad1))
  blocks.push(pixel(ax + 3, groundY - 2, 2, 1, pad2))
  blocks.push(pixel(ax + 3, groundY - 1, 2, 1, pad1))

  // crossbar
  blocks.push(pixel(ax, groundY - 5, 8, 1, post))

  // uprights
  blocks.push(pixel(ax,     groundY - 12, 2, 7, post))
  blocks.push(pixel(ax + 6, groundY - 12, 2, 7, post))

  return (
    // full-width staging area the size of the info box row
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: `${H * s}px`,
        margin: '0 auto'
      }}
    >
      {/* inner canvas that is pixel-sized, perfectly centered */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: 0,
          transform: 'translateX(-50%)',
          width: `${W * s}px`,
          height: `${H * s}px`
        }}
      >
        {blocks}
      </div>
    </div>
  )
}

function Timer({ seconds, running, onExpire }) {
  const [t, setT] = useState(seconds)

  useEffect(() => setT(seconds), [seconds])

  useEffect(() => {
    if (!running) return
    const start = Date.now()
    const id = setInterval(() => {
      const elapsed = Math.floor((Date.now() - start) / 1000)
      const remain = Math.max(0, seconds - elapsed)
      setT(remain)
      if (remain <= 0) {
        clearInterval(id)
        onExpire && onExpire()
      }
    }, 150)
    return () => clearInterval(id)
  }, [running, seconds, onExpire])

  return <div className="timer">{t}s</div>
}

export default function App() {
  const [screen, setScreen] = useState('opening')
  const [openingRunning, setOpeningRunning] = useState(true)
  const [questionRunning, setQuestionRunning] = useState(false) // hidden timer gate
  const [score, setScore] = useState(0)
  const [guess, setGuess] = useState('')
  const [actual, setActual] = useState(null)
  const [message, setMessage] = useState('')
  const [round, setRound] = useState(1)
  const [played, setPlayed] = useState(false)
  const [missed, setMissed] = useState(false)

  // NEW: weekend stat + submit flag
  const [fact, setFact] = useState(null)
  const [hasSubmitted, setHasSubmitted] = useState(false)

  const inputRef = useRef(null)
  const questionTimeoutRef = useRef(null) // hidden 30s timer

  // Demo fact pool (2 CFB + 3 NFL, 2025 season)
  const FACTS_2025 = [
    // CFB
    "CFB (2025): Missouri’s Ahmad Hardy led the FBS in rushing early with ~730 yards through September.",
    "CFB (2025): San José State’s Danny Scudero topped early receiving charts with ~665 yards.",
    // NFL
    "NFL (2025): Matthew Stafford opened the year leading the league with ~1,500 passing yards.",
    "NFL (2025): Jonathan Taylor surged to the early rushing lead with ~480 yards.",
    "NFL (2025): Puka Nacua paced early NFL receiving with ~580+ yards."
  ]

  useEffect(() => {
    // generate actual (and opening display) between 17 and 70 for demo
    setActual(17 + Math.floor(Math.random() * (70 - 17 + 1)))
  }, [round])

  function startQuestion() {
    if (played) return // only allow starting once
    setScreen('question')
    setOpeningRunning(false)
    setQuestionRunning(true)
    setPlayed(true)
    setGuess('')
    setMessage('')
    setFact(null)
    setHasSubmitted(false)
    setTimeout(() => inputRef.current && inputRef.current.focus(), 100)

    // start a hidden 30s timeout (no visible timer)
    if (questionTimeoutRef.current) clearTimeout(questionTimeoutRef.current)
    questionTimeoutRef.current = setTimeout(() => {
      // time up => compute correctness and show result
      endRound()
    }, 30000)
  }

  function endRound() {
    setQuestionRunning(false)
    if (questionTimeoutRef.current) {
      clearTimeout(questionTimeoutRef.current)
      questionTimeoutRef.current = null
    }

    // compute correctness at the end of the 30s window
    const g = parseInt(guess, 10)
    const correct = !isNaN(g) && g >= actual && g <= actual + 3

    setScreen('result')
    if (correct) {
      setScore((s) => s + 10)
      setMessage('Nice! +10 pts')
    } else {
      setMessage('Close but no — try again!')
    }
  }

  function handleOpeningExpire() {
    // called when the opening countdown reaches 0
    setOpeningRunning(false)
    setMissed(true)
    // show missed popup; don't auto-start the question
  }

  function resetDemo() {
    // allow trying again (demo only)
    setPlayed(false)
    setMissed(false)
    setScreen('opening')
    setOpeningRunning(true)
    setQuestionRunning(false)
    setGuess('')
    setMessage('')
    setFact(null)
    setHasSubmitted(false)
    // regenerate a new actual
    if (questionTimeoutRef.current) {
      clearTimeout(questionTimeoutRef.current)
      questionTimeoutRef.current = null
    }
    setActual(17 + Math.floor(Math.random() * (70 - 17 + 1)))
  }

  function submitGuess() {
    const g = parseInt(guess, 10)
    if (isNaN(g) || hasSubmitted) return

    // pick and show the stat *immediately* after answer; no result yet
    const pick = FACTS_2025[Math.floor(Math.random() * FACTS_2025.length)]
    setFact(pick)
    setHasSubmitted(true)
    setScreen('fact') // mid-screen shown until the 30s timer ends
  }

  return (
    <div className="app">
      <div className="game-window pixelated">
        {screen === 'opening' && (
          <div className="opening">
            <h1 className="title">Minigame!</h1>
            <div className="goal-wrap">
              <div className="scene pixel-scene"><PixelScene scale={2} /></div>
            </div>

            <div className="opening-info">
              {!missed && (
                <div className="field-good">
                  <div className="field-good-line">Field Goal was good from {actual} yards.</div>
                  <div className="field-good-line">How many yards back would it still be good from?</div>
                </div>
              )}
              {missed && (
                <div className="missed-popup">
                  <div className="missed-text">Your try is no good! Try again the next field goal</div>
                </div>
              )}
              <div className="opening-controls centered">
                <div className="countdown centered">
                  <div className="count-label">Play Game in:</div>
                  <Timer
                    seconds={20}               // 20s opening countdown
                    running={openingRunning}
                    onExpire={handleOpeningExpire}
                  />
                </div>

                <div className="opening-buttons centered">
                  {(!missed && !played) && (
                    <button className="btn play" onClick={startQuestion}>
                      PLAY
                    </button>
                  )}
                  {missed && (
                    <button className="btn tryagain" onClick={resetDemo}>
                      TRY AGAIN
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {screen === 'question' && (
          <div className="question">
            <div className="hud">
              {/* Timer intentionally hidden for this phase per requirements */}
            </div>

            <div className="prompt">
              <div className="prompt-text">
                <div className="prompt-line">Guess how far back the kick would still be</div>
                <div className="prompt-from">good from?</div>
              </div>
              <div className="input-row">
                <input
                  ref={inputRef}
                  className="guess-input"
                  value={guess}
                  onChange={(e) => setGuess(e.target.value.replace(/[^0-9]/g, ''))}
                  placeholder="yards (number)"
                />
                <button className="btn submit" onClick={submitGuess}>
                  Submit
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Interstitial showing the Weekend Stat while the hidden 30s runs */}
        {screen === 'fact' && (
          <div className="fact-screen">
            <div className="weekend-stat" role="status" aria-live="polite">
              <div className="weekend-stat__title">Weekend Stat</div>
              <div className="weekend-stat__body">{fact}</div>
            </div>
          </div>
        )}

        {screen === 'result' && (
          <div className="result">
            <h2>{message}</h2>
            <div className="result-detail">Actual: {actual} yards</div>
            <div className="result-detail">Your guess: {guess || '—'}</div>

            <div style={{ marginTop: 12, display: 'flex', justifyContent: 'center' }}>
              <button className="btn tryagain" onClick={resetDemo}>TRY AGAIN</button>
            </div>
          </div>
        )}
      </div>
      <footer className="credit">PrizePicks-styled Minigame</footer>
    </div>
  )
}
