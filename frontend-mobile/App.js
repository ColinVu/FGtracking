import React, { useEffect, useRef, useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  Pressable,
  TextInput,
  StyleSheet,
  Platform,
} from 'react-native';

// OPTIONAL: uncomment these 4 lines if you installed the font package
// import { useFonts, PressStart2P_400Regular } from '@expo-google-fonts/press-start-2p';

const COLORS = {
  electricViolet: '#8000FF',
  frenchGray: '#C7C6CE',
  kingfisher: '#3B047C',
  bg: '#050614',
  panel: '#0b0c17',
  btnTop: '#7a12ff',
  btnBottom: '#6100cc',
  title: '#a055ff',
  pink: '#FF5CA3',
  white: '#FFFFFF',
  black: '#000000',
};

/* -------------------- Pixel goalpost scene -------------------- */
const Pixel = ({ x, y, w = 1, h = 1, s, color }) => (
  <View
    style={{
      position: 'absolute',
      left: x * s,
      top: y * s,
      width: w * s,
      height: h * s,
      backgroundColor: color,
    }}
  />
);

function PixelScene({ scale = 2 }) {
  const s = 8 * scale;
  const W = 28;
  const H = 16;

  const post = '#FFD700';
  const pad1 = '#b62f4a';
  const pad2 = '#e24c66';
  const base = '#3b0f0f';

  const ax = Math.round((W - 8) / 2);
  const groundY = 12;

  const blocks = [
    <Pixel key="base" x={ax + 3} y={groundY + 1} w={2} h={1} s={s} color={base} />,
    <Pixel key="pole" x={ax + 3} y={groundY - 4} w={2} h={5} s={s} color={post} />,
    <Pixel key="band1" x={ax + 3} y={groundY - 3} w={2} h={1} s={s} color={pad1} />,
    <Pixel key="band2" x={ax + 3} y={groundY - 2} w={2} h={1} s={s} color={pad2} />,
    <Pixel key="band3" x={ax + 3} y={groundY - 1} w={2} h={1} s={s} color={pad1} />,
    <Pixel key="bar"  x={ax}     y={groundY - 5} w={8} h={1} s={s} color={post} />,
    <Pixel key="upr1" x={ax}     y={groundY - 12} w={2} h={7} s={s} color={post} />,
    <Pixel key="upr2" x={ax + 6} y={groundY - 12} w={2} h={7} s={s} color={post} />,
  ];

  return (
    <View style={{ width: '100%', height: H * s, alignItems: 'center', justifyContent: 'center' }}>
      <View style={{ width: W * s, height: H * s }}>
        {blocks}
      </View>
    </View>
  );
}

/* -------------------- tiny countdown hook -------------------- */
function useCountdown(seconds, running, onExpire) {
  const [t, setT] = useState(seconds);
  useEffect(() => setT(seconds), [seconds]);
  useEffect(() => {
    if (!running) return;
    const start = Date.now();
    const id = setInterval(() => {
      const elapsed = Math.floor((Date.now() - start) / 1000);
      const remain = Math.max(0, seconds - elapsed);
      setT(remain);
      if (remain <= 0) {
        clearInterval(id);
        onExpire && onExpire();
      }
    }, 150);
    return () => clearInterval(id);
  }, [running, seconds, onExpire]);
  return t;
}

/* --------------------------- App ---------------------------- */
export default function App() {
  // OPTIONAL font loading; uncomment if you installed @expo-google-fonts package
  // const [fontsLoaded] = useFonts({ PressStart2P_400Regular });
  // if (!fontsLoaded) return null;

  const [screen, setScreen] = useState('opening'); // opening | question | fact | result
  const [openingRunning, setOpeningRunning] = useState(true);
  const [questionRunning, setQuestionRunning] = useState(false); // hidden 30s timer
  const [score, setScore] = useState(0);
  const [guess, setGuess] = useState('');
  const [actual, setActual] = useState(null);
  const [message, setMessage] = useState('');
  const [played, setPlayed] = useState(false);
  const [missed, setMissed] = useState(false);

  const [fact, setFact] = useState(null);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const inputRef = useRef(null);
  const questionTimeoutRef = useRef(null);
  const submittedGuessRef = useRef(null);
  const submittedCorrectRef = useRef(null);

  const openingT = useCountdown(20, openingRunning, () => {
    setOpeningRunning(false);
    setMissed(true);
  });

  const FACTS_2025 = [
    'CFB (2025): Missouri’s Ahmad Hardy led the FBS in rushing early with ~730 yards through September.',
    'CFB (2025): San José State’s Danny Scudero topped early receiving charts with ~665 yards.',
    'NFL (2025): Matthew Stafford opened the year leading the league with ~1,500 passing yards.',
    'NFL (2025): Jonathan Taylor surged to the early rushing lead with ~480 yards.',
    'NFL (2025): Puka Nacua paced early NFL receiving with ~580+ yards.',
  ];

  // re-seed the "actual" distance when we enter opening
  useEffect(() => {
    if (screen === 'opening') {
      setActual(17 + Math.floor(Math.random() * (70 - 17 + 1)));
    }
  }, [screen]);

  function startQuestion() {
    if (played) return;
    setScreen('question');
    setOpeningRunning(false);
    setQuestionRunning(true);
    setPlayed(true);
    setGuess('');
    setMessage('');
    setFact(null);
    setHasSubmitted(false);
    setTimeout(() => inputRef.current?.focus(), 100);

    if (questionTimeoutRef.current) clearTimeout(questionTimeoutRef.current);
    // clear any previous submitted value when a new question starts
    submittedGuessRef.current = null;
    submittedCorrectRef.current = null;
    questionTimeoutRef.current = setTimeout(() => {
      endRound(); // compute result only when 30s expires
    }, 30000);
  }

  function endRound() {
    setQuestionRunning(false);
    if (questionTimeoutRef.current) {
      clearTimeout(questionTimeoutRef.current);
      questionTimeoutRef.current = null;
    }
  // prefer the authoritative submitted value (if any)
  const g = submittedGuessRef.current ?? parseInt(guess, 10);
  console.log('endRound: guess=', guess, 'g=', g, 'actual=', actual, 'hasSubmitted=', hasSubmitted, 'submittedCorrectRef=', submittedCorrectRef.current);
  const correct = submittedCorrectRef.current ?? (!isNaN(g) && actual !== null && g >= actual && g <= actual + 3);

    setScreen('result');
    if (correct) {
      setScore((s) => s + 10);
      setMessage('Nice! +10 pts');
    } else {
      setMessage('Close but no — try again!');
    }

    // clear stored submitted values for next round
    submittedGuessRef.current = null;
    submittedCorrectRef.current = null;
  }

  function resetDemo() {
    setPlayed(false);
    setMissed(false);
    setScreen('opening');
    setOpeningRunning(true);
    setQuestionRunning(false);
    setGuess('');
    setMessage('');
    setFact(null);
    setHasSubmitted(false);
    if (questionTimeoutRef.current) {
      clearTimeout(questionTimeoutRef.current);
      questionTimeoutRef.current = null;
    }
    setActual(17 + Math.floor(Math.random() * (70 - 17 + 1)));
    submittedGuessRef.current = null;
    submittedCorrectRef.current = null;
  }

  function submitGuess() {
    const g = parseInt(guess, 10);
    if (isNaN(g) || hasSubmitted) return;

    // stash parsed numeric answer immediately and precompute correctness
    submittedGuessRef.current = g;
    submittedCorrectRef.current = (g >= actual && g <= actual + 3);
    console.log('submitGuess: guess=', guess, 'parsed=', g, 'actual=', actual, 'submittedCorrect=', submittedCorrectRef.current);

    const pick = FACTS_2025[Math.floor(Math.random() * FACTS_2025.length)];
    setFact(pick);
    setHasSubmitted(true);
    setScreen('fact'); // show stat immediately; result waits for hidden timer
  }

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.window}>
        {screen === 'opening' && (
          <View>
            <Text style={[styles.title /* , { fontFamily: 'PressStart2P_400Regular' } */]}>
              Minigame!
            </Text>

            <View style={styles.sceneWrap}>
              <PixelScene scale={2} />
            </View>

            <View style={{ marginTop: 8 }}>
              {!missed ? (
                <View>
                  <Text style={styles.fieldLine}>
                    Field Goal was good from {actual ?? '--'} yards.
                  </Text>
                  <Text style={styles.fieldLine}>
                    How many yards back would it still be good from?
                  </Text>
                </View>
              ) : (
                <Text style={[styles.fieldLine, { color: COLORS.white, textAlign: 'center' }]}>
                  Your try is no good! Try again the next field goal
                </Text>
              )}

              <View style={styles.openingControls}>
                <View style={{ alignItems: 'center' }}>
                  <Text style={styles.countLabel}>Play Game in:</Text>
                  <View style={styles.timer}>
                    <Text style={styles.timerText}>{openingT}s</Text>
                  </View>
                </View>

                {!missed && !played ? (
                  <Pressable style={styles.btn} onPress={startQuestion}>
                    <Text style={styles.btnText}>PLAY</Text>
                  </Pressable>
                ) : missed ? (
                  <Pressable style={styles.btn} onPress={resetDemo}>
                    <Text style={styles.btnText}>TRY AGAIN</Text>
                  </Pressable>
                ) : null}
              </View>
            </View>
          </View>
        )}

        {screen === 'question' && (
          <View>
            {/* hidden 30s timer by design */}
            <View style={styles.promptBox}>
              <View style={{ marginBottom: 12 }}>
                <Text style={styles.promptLine}>Guess how far back the kick would still be</Text>
                <Text style={[styles.promptLine, { fontWeight: '700' }]}>good from?</Text>
              </View>

              <View style={styles.inputRow}>
                <TextInput
                  ref={inputRef}
                  value={guess}
                  onChangeText={(t) => setGuess(t.replace(/[^0-9]/g, ''))}
                  placeholder="yards (number)"
                  placeholderTextColor="rgba(255,255,255,0.45)"
                  keyboardType="number-pad"
                  style={styles.input}
                  returnKeyType="done"
                  onSubmitEditing={submitGuess}
                />
                <Pressable style={styles.btn} onPress={submitGuess}>
                  <Text style={styles.btnText}>Submit</Text>
                </Pressable>
              </View>
            </View>
          </View>
        )}

        {screen === 'fact' && (
          <View style={styles.factScreen}>
            <View style={styles.statCard} accessibilityRole="summary">
              <Text style={styles.statTitle}>Weekend Stat</Text>
              <Text style={styles.statBody}>{fact}</Text>
            </View>
          </View>
        )}

        {screen === 'result' && (
          <View style={{ alignItems: 'center' }}>
            <Text style={styles.resultTitle}>{message}</Text>
            <Text style={styles.resultDetail}>Actual: {actual ?? '--'} yards</Text>
            <Text style={styles.resultDetail}>Your guess: {guess || '—'}</Text>

            <Pressable style={[styles.btn, { marginTop: 16 }]} onPress={resetDemo}>
              <Text style={styles.btnText}>TRY AGAIN</Text>
            </Pressable>
          </View>
        )}
      </View>

      <Text style={styles.credit}>PrizePicks-styled Minigame</Text>
    </SafeAreaView>
  );
}

/* --------------------------- Styles --------------------------- */
const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  window: {
    width: 720,
    maxWidth: '95%',
    backgroundColor: COLORS.panel,
    borderColor: COLORS.btnTop,
    borderWidth: 8,
    padding: 20,
    shadowColor: COLORS.black,
    shadowOpacity: 0.6,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
    borderRadius: 8,
  },
  title: {
    fontSize: 24,
    color: COLORS.title,
    textAlign: 'center',
    // fontFamily: 'PressStart2P_400Regular',
  },
  sceneWrap: {
    marginVertical: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fieldLine: {
    fontSize: 14,
    color: COLORS.frenchGray,
    textAlign: 'center',
    marginBottom: 8,
  },
  openingControls: {
    marginTop: 12,
    alignItems: 'center',
    gap: 12, // RN supports gap on new RN versions; if it warns, replace with margin
  },
  countLabel: { fontSize: 14, color: COLORS.title, marginBottom: 6 },
  timer: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  timerText: { color: COLORS.title, fontWeight: '700' },

  btn: {
    backgroundColor: COLORS.btnTop,
    paddingVertical: 14,
    paddingHorizontal: 22,
    borderRadius: 8,
    ...Platform.select({
      ios: { shadowColor: COLORS.black, shadowOpacity: 0.45, shadowRadius: 8, shadowOffset: { width: 0, height: 8 } },
      android: { elevation: 6 },
    }),
  },
  btnText: { color: COLORS.white, fontWeight: '800', textAlign: 'center' },

  promptBox: {
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.06)',
    padding: 14,
    alignItems: 'center',
  },
  promptLine: { fontSize: 12, textAlign: 'center', marginBottom: 8, color: COLORS.frenchGray },

  inputRow: { flexDirection: 'row', gap: 10, alignItems: 'center' },
  input: {
    padding: 10,
    fontSize: 14,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.06)',
    backgroundColor: '#0d1630',
    color: COLORS.frenchGray,
    width: 160,
    borderRadius: 6,
  },

  factScreen: { alignItems: 'center', justifyContent: 'center', padding: 16 },
  statCard: {
    marginTop: 16,
    maxWidth: 560,
    paddingVertical: 16,
    paddingHorizontal: 20,
    backgroundColor: COLORS.btnBottom,   // purple background
    borderColor: COLORS.frenchGray,      // gray border (as requested)
    borderWidth: 2,
    borderRadius: 12,
    alignItems: 'center',
  },
  statTitle: { color: COLORS.white, fontWeight: '900', marginBottom: 6, fontSize: 16, textAlign: 'center' },
  statBody:  { color: COLORS.white, fontWeight: '700', textAlign: 'center', lineHeight: 20 },

  resultTitle: { fontSize: 22, color: COLORS.title, textAlign: 'center', marginBottom: 8 },
  resultDetail: { fontSize: 12, color: COLORS.frenchGray, textAlign: 'center', marginTop: 6 },

  credit: { marginTop: 10, fontSize: 11, color: 'rgba(255,255,255,0.5)' },
});
