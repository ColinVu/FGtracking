import React, { useState, useEffect, useRef } from 'react'

const PRIZEPICKS_COLORS = {
  purple: '#5B2E8A',
  pink: '#FF5CA3',
  dark: '#0F1226',
  light: '#F7F7FB',
  accent: '#00D1B2'
}

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
        width: '100%',           // take the full width of the panel row
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
  }, [running, seconds])

  return <div className="timer">{t}s</div>
}

export default function App() {
  const [screen, setScreen] = useState('opening')
  const [openingRunning, setOpeningRunning] = useState(true)
  const [questionRunning, setQuestionRunning] = useState(false)
  const [score, setScore] = useState(0)
  const [guess, setGuess] = useState('')
  const [actual, setActual] = useState(null)
  const [message, setMessage] = useState('')
  const [round, setRound] = useState(1)
  const [played, setPlayed] = useState(false)
  const [missed, setMissed] = useState(false)
  const inputRef = useRef(null)

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
    setTimeout(() => inputRef.current && inputRef.current.focus(), 100)
  }

  function endRound(correct) {
    setQuestionRunning(false)
    setScreen('result')
    if (correct) {
      setScore((s) => s + 10)
      setMessage('Nice! +10 pts')
    } else {
      setMessage('Close but no — try again!')
    }
    // Do not auto-restart. Game runs only once per requirement.
    // Keep result on screen and do not schedule another round.
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
    // regenerate a new actual
    setActual(17 + Math.floor(Math.random() * (70 - 17 + 1)))
  }

  function submitGuess() {
    const g = parseInt(guess, 10)
    if (isNaN(g)) return
    const isCorrect = g >= actual && g <= actual + 3
    endRound(isCorrect)
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
                    seconds={10}
                    running={openingRunning}
                    onExpire={() => {
                      handleOpeningExpire()
                    }}
                  />
                </div>

                <div className="opening-buttons centered">
                  {(!missed && !played) && (
                    <button className="btn play" onClick={() => startQuestion()}>
                      PLAY
                    </button>
                  )}
                  {missed && (
                    <button className="btn tryagain" onClick={() => resetDemo()}>
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
              <Timer
                seconds={7}
                running={questionRunning}
                onExpire={() => {
                  endRound(false)
                }}
              />
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
              {/* hint removed for production - actual value hidden */}
            </div>
          </div>
        )}

        {screen === 'result' && (
          <div className="result">
            <h2>{message}</h2>
            <div className="result-detail">Actual: {actual} yards</div>
            <div className="result-detail">Your guess: {guess || '—'}</div>
            <div style={{ marginTop: 12, display: 'flex', justifyContent: 'center' }}>
              <button className="btn tryagain" onClick={() => resetDemo()}>TRY AGAIN</button>
            </div>
          </div>
        )}
      </div>
      <footer className="credit">PrizePicks-styled Minigame</footer>
    </div>
  )
}
