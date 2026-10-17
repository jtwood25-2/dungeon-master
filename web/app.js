// Dungeon Master phone client: capture mic -> 16k PCM -> PC; play DM audio; show latency.
const TARGET_SR = 16000;
const UPLINK_ULAW = true;   // halve the upstream; set false to send raw PCM
let micFrames = 0, micSent = 0;                 // what the audio graph actually produced
const micSkips = { notCapturing: 0, socket: 0 };
let ws, ctx, source, processor, stream, recording = false, reconnects = 0;
const audioQ = [];           // queued MP3 blobs to play in order
let playing = false;

// The server can see a socket close but never WHY. A reload, a tab iOS decided to discard,
// a frozen page and a dropped link are indistinguishable from the other end, and they have
// completely different fixes -- so the phone says which one it was.
function report(event, detail) {
  try {
    const body = JSON.stringify({ event, detail: String(detail || "") });
    // keepalive/sendBeacon so it still goes out while the page is being torn down
    if (navigator.sendBeacon) navigator.sendBeacon("/clientlog", new Blob([body], { type: "application/json" }));
    else fetch("/clientlog", { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true });
  } catch (e) {}
}
const el = (id) => document.getElementById(id);
const conn = el("conn"), talk = el("talk"), log = el("log");

function addMsg(cls, html) {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.innerHTML = html;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}

// ---- websocket ----
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    conn.textContent = "online"; conn.className = "pill on";
    talk.disabled = false; hf.disabled = false;
    if (reconnects) { addMsg("sys", "reconnected — carry on"); checkBuild(); }
    reconnects = 0;
    // the server opens on its own best guess; tell it which one we're on
    ws.send(JSON.stringify({ type: "uplink", codec: UPLINK_ULAW ? "ulaw" : "pcm16" }));
    if (current) ws.send(JSON.stringify({ type: "campaign", name: current }));
    refreshStart();
    // hands-free survives the blip: re-arm it rather than making them tap again
    if (handsfree) ws.send(JSON.stringify({ type: "listen" }));
  };
  // A dropped socket is usually the server picking up a new build, not a crash. Reconnect
  // quickly and quietly, keep the log, and say "reconnecting" rather than "offline" so it
  // doesn't read as the app dying mid-session.
  ws.onclose = (e) => {
    report("ws-close", `code=${e.code} reason=${e.reason || "-"} clean=${e.wasClean} speaking=${speechPlaying}`);
    conn.textContent = reconnects ? "reconnecting…" : "offline";
    conn.className = "pill off";
    talk.disabled = true; hf.disabled = true;
    reconnects++;
    setTimeout(connect, Math.min(800 * reconnects, 4000));
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) { enqueueAudio(e.data); return; }
    const m = JSON.parse(e.data);
    if (m.type === "ambience") { setAmbience(m.audio, m.mood); return; }
    if (m.type === "sfx") { playSfx(m.audio, m.name); return; }
    if (m.type === "map") {
      // a quiet update is a token move or a newly revealed room: refresh in place, never
      // pop the panel open over what they're doing
      const open = !m.quiet && turnOver && !speechPlaying;
      showMap(m.title, m.svg, open);
      if (!m.quiet) addMsg("sys", "🗺 " + m.title + " — tap 🗺 Map to view");
      return;
    }
    if (m.type === "roster") {
      ROSTER = m.characters || [];
      MY_SID = m.sid || MY_SID;
      // Re-claim the seat this device had before a reload or a dropped tunnel, so a phone
      // coming back does not silently become "nobody" in the middle of a fight.
      // ONCE per connection, and never over the top of somebody else. One phone sent three
      // claims in three seconds tonight -- Oktah, Merengue, Oktah -- and because sit() hands a
      // character to the newest claimant, a reload could knock another player out of their
      // seat. The server tells us who holds what; re-claim only a seat that is genuinely free.
      if (!_reclaimed) {
        _reclaimed = true;
        let want = MY_SEAT;
        try { want = want || localStorage.getItem("dm.seat") || ""; } catch (e) {}
        const taken = (LAST_SEATS || []).some(
          (s) => s.character && s.sid !== MY_SID &&
                 s.character.toLowerCase() === (want || "").toLowerCase());
        if (want && !taken && ws && ws.readyState === 1) {
          ws.send(JSON.stringify({ type: "seat", character: want }));
        } else if (taken) {
          addMsg("sys", want + " is already on another phone — tap 🪑 to pick");
        }
      }
      return;
    }
    if (m.type === "table") { showTable(m); return; }
    if (m.type === "turnwait") { showEndTurn(m.who || ""); return; }
    if (m.type === "combat") {
      showInitiative(m);
      // whose turn it is comes with every initiative update, so the button follows the order
      // without needing its own message
      showEndTurn(m.in_combat ? (m.turn || "") : "");
      return;
    }
    if (m.type === "turn_start") { el("startBar").hidden = true; turnOver = false; setHfState("thinking…"); showStop(true); return; }
    if (m.type === "turn_end") {
      turnOver = true; if (!speechPlaying) setHfState("listening");
      if (HALF_DUPLEX && handsfree && micHeld) scheduleMicReopen(400);
      return;
    }
    // barge-in confirmed: the old answer goes quiet, but a new turn is already under way
    if (m.type === "interrupted") { stopSpeech(); turnOver = false; setHfState("thinking…"); return; }
    if (m.type === "feedback") {                 // captured mid-play; never reaches the DM
      addMsg("sys", "📝 " + m.item.text);
      const n = el("notesCount"); n.textContent = " " + ((parseInt(n.textContent) || 0) + 1);
      if (!notesPanel.hidden) loadNotes();
      return;
    }
    if (m.type === "transcript") {
      addMsg("you", (m.speaker ? m.speaker + ": " : "") + m.text);
      el("l-stt").textContent = m.ms + "ms";
    }
    else if (m.type === "dm")    { addMsg("dm", "<b>DM:</b> " + m.text); }
    else if (m.type === "status")   { addMsg("sys", m.msg);
      if (m.msg === "listening") setHfState("listening");
      if (/learned/i.test(m.msg)) enrollPanel.hidden = true; }   // voice saved -> collapse panel
    else if (m.type === "timing") {
      if (m.llm_first_token_ms != null) el("l-llm").textContent = m.llm_first_token_ms + "ms";
      if (m.first_audio_ms != null)     el("l-audio").textContent = m.first_audio_ms + "ms";
    }
  };
}

// Playback goes through HTMLAudioElement (NOT Web Audio): media elements play through the
// iPhone silent/ring switch, whereas Web Audio output is muted by it. ctx is kept only for
// the mic-capture path. The element is primed with a silent clip inside the first tap.
function ensureCtx() {
  if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) {} }
  return ctx;
}
function beepWav(freq, ms, sr = 8000) {             // small WAV data-URI (freq 0 = silent)
  const n = Math.floor(sr * ms / 1000), d = new DataView(new ArrayBuffer(44 + n * 2));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) d.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); d.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
  d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
  d.setUint32(24, sr, true); d.setUint32(28, sr * 2, true); d.setUint16(32, 2, true);
  d.setUint16(34, 16, true); w(36, "data"); d.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) d.setInt16(44 + i * 2, Math.sin(2 * Math.PI * freq * i / sr) * 0.3 * 32767, true);
  let b = ""; const u = new Uint8Array(d.buffer);
  for (let i = 0; i < u.length; i++) b += String.fromCharCode(u[i]);
  return "data:audio/wav;base64," + btoa(b);
}
// A long intro is 30-60s where the player is listening and touching nothing. That is exactly
// when iOS dims, then sleeps, then freezes or discards the page -- taking the websocket and
// the rest of the narration with it. Hold a wake lock while the DM has the floor and drop it
// the moment it stops, so nothing is held while they are just thinking.
let _wakeLock = null;
let _wakeWanted = false;
async function holdWake() {
  _wakeWanted = true;
  // NotAllowedError: iOS only grants this inside a user gesture and while visible, so asking
  // from the audio callback failed every single time. Ask on the tap instead, and re-ask on
  // the way back from hidden -- both are moments the browser accepts.
  if (document.visibilityState !== "visible") return;
  try {
    if (!_wakeLock && navigator.wakeLock) {
      _wakeLock = await navigator.wakeLock.request("screen");
      _wakeLock.addEventListener("release", () => { _wakeLock = null; });
    }
  } catch (e) { /* not granted here; the gesture path will get it */ }
}
function dropWake() {
  _wakeWanted = false;
  try { if (_wakeLock) { _wakeLock.release(); _wakeLock = null; } } catch (e) {}
}
let audioUnlocked = false, _audioDiag = false, speechEl = null, ambientEl = null;
// 0.16 was inaudible under speech; loud enough to feel, quiet enough to talk over
// Multiplied by the loop's own peak on the server (ambience.GAIN), so the audible level
// is the PRODUCT of the two -- which is how this ended up loud with neither number
// looking wrong on its own. A bed you have to listen for is the only kind that survives
// a whole session under a narrator.
const AMBIENCE_VOLUME = 0.22;
const speechQ = []; let speechPlaying = false;
// The mic never stops. The server only needs to know whether the speaker is live, so it can
// demand a clearer burst of speech before treating it as a barge-in rather than as echo.
let turnOver = true;
function reportPlaying(on) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "playing", on }));
}

// Barge-in: cut the DM off mid-sentence when it gets something wrong. Playback is killed
// here rather than waiting on the server, so it goes quiet the instant you tap.
function interruptDM() {
  stopSpeech();
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "interrupt" }));
  setHfState("listening"); showStop(false);
}
function stopSpeech() {
  speechQ.length = 0;
  if (speechEl) { try { speechEl.pause(); } catch (e) {} try { speechEl.currentTime = 0; } catch (e) {} }
  speechPlaying = false; turnOver = true; _quietSince = Date.now(); dropWake();
  reportPlaying(false);
  // Stopping the DM is also a request to be heard again -- without this, ⏹ in half-duplex left
  // the microphone shut with nothing left to trigger the handover.
  if (HALF_DUPLEX && handsfree && micHeld) scheduleMicReopen(250);
}
function showStop(on) { const b = el("stop"); if (b) b.hidden = !on; }
function getSpeechEl() {
  if (!speechEl) {
    speechEl = new Audio(); speechEl.setAttribute("playsinline", "");
    speechEl.addEventListener("ended", _nextSpeech);
    speechEl.addEventListener("error", _nextSpeech);
  }
  return speechEl;
}
function unlockAudio() {
  if (!audioUnlocked) {                              // prime the media element inside the gesture
    audioUnlocked = true;
    const el = getSpeechEl(); el.src = beepWav(0, 40);
    el.play().then(() => el.pause()).catch(() => {});
    const amb = getAmbientEl(); amb.src = beepWav(0, 40);   // prime it too, or iOS blocks it
    amb.play().then(() => amb.pause()).catch(() => {});
  }
  ensureCtx(); if (ctx && ctx.state !== "running") ctx.resume();   // (for the mic only)
}
// Two extra roles, chosen by URL, for running a table of seven off one room speaker:
//
//   ?speaker   play the DM out loud and never touch the microphone   (open this on the PC,
//              which is what the Bluetooth speaker should be paired to)
//   ?mic       capture only, never play a sound                      (the phone on the table)
//
// Splitting them is not a convenience. A phone doing OUTPUT and INPUT at once is what makes
// iOS reconfigure its audio session, and reconfiguring the session is what ends the capture
// track -- "mic is open but hearing nothing", every time the speaker came alive.
// One definition of "not the phone's own microphone", used when pinning the input, when
// checking mid-session whether iOS has rerouted it, and when deciding half-duplex.
//
// It asks whether the device IS the phone, not whether its name is on a list of gadgets.
// The list was the bug: an Anker PowerConf offered itself as "Anker PowerConf" and matched
// none of bluetooth / headset / speaker / usb, so a conference speakerphone sitting on the
// table was classified as "no bluetooth input" and the phone's own microphone was chosen
// instead. Every device has a brand name and no list will ever have them all; there is only
// ever one built-in microphone, and it says so.
const BUILT_IN_RE = /iphone|ipad|built-?in|internal|^default$/i;
// Kept only as a POSITIVE hint for ranking, never as the test for external.
const NOT_BUILT_IN_RE = /bluetooth|hands-?free|headset|airpod|buds|speaker|usb|powerconf/i;
const isExternalMic = (label) => !!label && !BUILT_IN_RE.test(label);
const ROLE = new URLSearchParams(location.search);
const SPEAKER_ONLY = ROLE.has("speaker");
// ?mic=phone forces the phone's own microphone (best capture, no Bluetooth output);
// ?mic=bluetooth forces the speaker's (worse capture, the room hears the DM);
// anything else auto-detects, which prefers Bluetooth when it is actually there.
const MIC_PREF = (ROLE.get("mic") || "auto").toLowerCase();
// ?duplex=half closes the microphone while the DM speaks, so iOS drops back to its
// Playback category and a Bluetooth speaker can have the audio. Costs barge-in.
//
// Not a constant, because it is a WORKAROUND FOR A SPEAKER WITH NO MICROPHONE and it stops
// being the right answer the moment one is plugged in. A speaker that is output-only leaves
// iOS an impossible choice -- PlayAndRecord keeps the mic and loses A2DP, Playback keeps the
// speaker and kills the mic -- and closing the mic for playback is the only way through it.
// A speakerphone presents BOTH directions, so iOS puts the session on a duplex route and
// carries them at once; holding the mic shut there buys nothing and costs barge-in, costs the
// device's own echo canceller its reference, and makes a locked screen deafen the session.
// So: honour the URL as an opening position, then let the hardware overrule it.
let HALF_DUPLEX = (ROLE.get("duplex") || "").toLowerCase() === "half";
// ?role=mic makes this device capture only and never make a sound (its partner is a
// ?speaker page). Distinct from ?mic=, which chooses WHICH microphone.
const MIC_ONLY = ROLE.get("role") === "mic";

function enqueueAudio(buf) {
  if (MIC_ONLY) return;                 // the room speaker has this; stay silent
  // sniff the container so either voice engine works: local kokoro sends WAV, edge-tts MP3
  const h = new Uint8Array(buf, 0, 4);
  const isWav = h[0] === 0x52 && h[1] === 0x49 && h[2] === 0x46 && h[3] === 0x46;  // "RIFF"
  speechQ.push(URL.createObjectURL(new Blob([buf], { type: isWav ? "audio/wav" : "audio/mpeg" })));
  if (!speechPlaying) _nextSpeech();
}
// -- half-duplex mic handover ------------------------------------------------
// Closing the mic is what lets a Bluetooth speaker have the audio; the whole trick is being
// careful about WHEN it comes back.
let _reopenTimer = 0;
// WHO IS HOLDING THE MICROPHONE DOWN, and since when. `speechPlaying` was standing in for this
// and it is not the same thing: the DM streams a sentence at a time, so the queue runs dry
// between sentences and speechPlaying goes false in every gap. Every guard keyed on it therefore
// unlatched two or three times a turn, and the watchdog "rescued" the mic in the gap -- which is
// the audio hopping back to the phone mid-narration. This flag is set once when the mic is shut
// and cleared once when it is actually back, so the gaps cannot unlatch anything.
// _heldSince is when the mic was shut and only ever reports; _quietSince is when audio last
// STOPPED, and it is the one the fallback deadline measures. Confusing the two would arm the
// deadline against the length of the narration -- any turn longer than the timeout would hand
// the mic back in the middle of itself.
let micHeld = false, _heldSince = 0, _quietSince = 0, _reopenTries = 0;
async function closeMicForPlayback() {
  capturing = false;
  micHeld = true; _heldSince = Date.now(); _quietSince = 0; _reopenTries = 0;
  closeMic(); source = null; processor = null;
  // Stopping the capture track is NOT enough, which the log settled: the microphone was shut
  // for thirty-three seconds and the narration still came out of the phone. A live
  // AudioContext keeps the iOS audio session in a recording-capable category on its own, and
  // A2DP is not offered in that category -- so the context has to go quiet too. Suspend
  // rather than close: a closed one cannot be revived without a user gesture, and that would
  // cost the microphone entirely.
  let st = "no-ctx";
  try {
    if (ctx && ctx.state === "running") { await ctx.suspend(); st = ctx.state; }
    else if (ctx) st = ctx.state;
  } catch (e) { st = "suspend-failed:" + (e && e.name); }
  report("duplex", `mic closed for playback (ctx=${st})`);
}
function cancelMicReopen() { if (_reopenTimer) { clearTimeout(_reopenTimer); _reopenTimer = 0; } }

// Give the microphone back. The one place that undoes closeMicForPlayback, so there is exactly
// one path back and every caller reports the same way.
async function releaseMic(why) {
  cancelMicReopen();
  try { if (ctx && ctx.state === "suspended") await ctx.resume(); } catch (e) {}
  const ok = await openMic();
  if (ok) {
    micHeld = false; _reopenTries = 0;
    capturing = !!handsfree;
    report("duplex", `mic reopened after ${((Date.now() - _heldSince) / 1000).toFixed(1)}s `
      + `(${why}, ctx=${ctx && ctx.state})`);
  } else {
    _reopenTries++;
    report("duplex", `mic reopen FAILED (${why}, try ${_reopenTries}, `
      + `page=${document.visibilityState}, ctx=${ctx && ctx.state})`);
    // Backgrounded is not a failure, it is a wait: iOS will not hand a hidden page a capture
    // device at all. Say so once rather than counting it against the retries forever.
    if (_reopenTries === 4 && document.visibilityState === "visible") {
      addMsg("sys", "mic didn't come back — tap 🎙️ Auto off and on");
    }
    scheduleMicReopen(700);
  }
  return ok;
}

function scheduleMicReopen(delay = 700) {
  cancelMicReopen();
  _reopenTimer = setTimeout(async () => {
    _reopenTimer = 0;
    if (!HALF_DUPLEX || !handsfree) return;
    if (speechPlaying || speechQ.length) { scheduleMicReopen(400); return; }
    // turn_end is the clean signal that no more audio is coming, and normally it is the one to
    // wait for. But it is a message from the server, and if it is late, dropped, or the turn
    // errored, waiting for it forever means the microphone never comes back and the session is
    // deaf from the first reply onward -- which is exactly what "it stops after the first
    // message" was. Silence for four seconds says the same thing the message would have.
    const quietFor = Date.now() - (_quietSince || Date.now());
    if (!turnOver && quietFor < 4000) { scheduleMicReopen(400); return; }
    // A hidden page cannot be given a microphone; wait for it to come back rather than burning
    // retries and giving up while the phone is in somebody's pocket.
    if (document.visibilityState !== "visible") { scheduleMicReopen(600); return; }
    await releaseMic(turnOver ? "turn ended" : `no audio for ${(quietFor / 1000).toFixed(1)}s`);
  }, delay);
}

function _nextSpeech() {
  const el = getSpeechEl();
  if (el._u) { URL.revokeObjectURL(el._u); el._u = null; }
  if (!speechQ.length) {
    speechPlaying = false; _quietSince = Date.now(); showStop(false); reportPlaying(false); dropWake();
    // HALF-DUPLEX: put the microphone back -- but NOT the instant the queue empties.
    // The DM streams a sentence at a time, so the queue runs dry between sentences, and
    // reopening there flipped iOS straight back into PlayAndRecord mid-turn. The audio for
    // the next sentence then came out of the phone. Wait for the turn to be over AND for a
    // quiet gap, and let any new clip cancel the reopen.
    if (HALF_DUPLEX && handsfree) scheduleMicReopen();
    setHfState("listening");
    return;
  }
  if (!speechPlaying) reportPlaying(true);
  // An open microphone forces iOS into its PlayAndRecord audio category, and A2DP is simply
  // not offered in that category -- so while we are capturing, a Bluetooth speaker cannot be
  // the output and the DM comes out of the phone. Measured on this phone: the speaker does
  // not appear as an audio INPUT either, so the hands-free profile that would carry both
  // directions is not available. Nothing can route Bluetooth output while the mic is open.
  //
  // But nothing needs the mic open WHILE the DM is talking. Close it for the duration, let
  // iOS fall back to Playback and hand the audio to the speaker, and reopen the moment it
  // stops. The cost is that you cannot interrupt mid-sentence; the gain is that the room
  // hears the DM at all.
  // Close once per turn, not once per sentence: re-closing an already-held mic would restart
  // both clocks every clip and the fallback deadline would never come due.
  if (HALF_DUPLEX) { cancelMicReopen(); if (!micHeld) closeMicForPlayback(); }
  speechPlaying = true; setHfState("speaking"); holdWake();
  const url = speechQ.shift(); el._u = url; el.src = url;
  el.play().then(() => { if (!_audioDiag) { _audioDiag = true; addMsg("sys", "♪ audio playing"); } })
           .catch((e) => { if (!_audioDiag) { _audioDiag = true; addMsg("sys", "⚠ playback blocked: " + e); } _nextSpeech(); });
}
// iOS refuses to play an Audio element that wasn't created and started inside a user gesture,
// and the rejection was being swallowed — so the background bed simply never existed. One
// element is primed during the first tap (see unlockAudio) and its source swapped after.
function getAmbientEl() {
  if (!ambientEl) {
    ambientEl = new Audio();
    ambientEl.setAttribute("playsinline", "");
    ambientEl.loop = true;
    ambientEl.volume = AMBIENCE_VOLUME;
  }
  return ambientEl;
}
function setAmbience(dataUrl, mood) {
  const a = getAmbientEl();
  if (!dataUrl) { a.pause(); return; }
  a.src = dataUrl;
  a.volume = AMBIENCE_VOLUME;
  a.play().then(() => addMsg("sys", "🎵 " + (mood || "ambience")))
          .catch((e) => addMsg("sys", "🎵 " + (mood || "ambience") + " (blocked: tap the screen)"));
}

// One-shots play over the ambience loop rather than replacing it, and are deliberately
// louder -- they are punctuation on a dramatic beat, not background.
const SFX_VOLUME = 0.55;
function playSfx(dataUrl, name) {
  if (!dataUrl) return;
  const s = new Audio(dataUrl);
  s.volume = SFX_VOLUME;
  s.play().catch(() => {});          // autoplay refusal must never break the turn
  addMsg("sys", "▶ " + (name || "sfx"));
}

// ---- mic capture ----
function downsample(f32, inRate) {
  if (inRate === TARGET_SR) return f32;
  const ratio = inRate / TARGET_SR, out = new Float32Array(Math.floor(f32.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio), end = Math.floor((i + 1) * ratio);
    let s = 0, n = 0;
    for (let j = start; j < end && j < f32.length; j++) { s += f32[j]; n++; }
    out[i] = n ? s / n : 0;
  }
  return out;
}
// The mic streams CONTINUOUSLY so a player can answer the instant the DM stops. At 16kHz
// 16-bit that is ~32KB/s of sustained upload for the whole session, on the direction a phone
// is weakest and through a tunnel that has been dropping mid-narration. mu-law halves it for
// about a third of a dB of speech quality, and every browser can do it.
function toUlaw(i16) {
  const out = new Uint8Array(i16.length);
  for (let i = 0; i < i16.length; i++) {
    let s = i16[i];
    const sign = s < 0 ? 0x80 : 0;
    if (s < 0) s = -s;
    if (s > 32635) s = 32635;
    s += 132;                                  // bias
    let e = 7;
    for (let m = 0x4000; (s & m) === 0 && e > 0; m >>= 1) e--;
    const mantissa = (s >> (e + 3)) & 0x0F;
    out[i] = ~(sign | (e << 4) | mantissa) & 0xFF;
  }
  return out;
}
function toInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); out[i] = s * 32767; }
  return out;
}

let handsfree = false, capturing = false;
let micPeak = 0, micLevel = 0, micTimer = 0;   // loudest sample, smoothed level, reporter
const hf = el("handsfree");

// Hands-free is half-duplex, so show which half we're in -- "is it even listening to me?"
// is otherwise invisible, and the answer decides whether the player should just talk.
const HF_LABEL = { listening: "🎙️ Listening", "thinking…": "💭 Thinking…", speaking: "🔊 DM" };
function setHfState(s) {
  if (!handsfree) { hf.textContent = "🎙️ Auto"; return; }
  hf.textContent = HF_LABEL[s] || "🎙️ Auto ✓";
  hf.classList.toggle("busy", s !== "listening");
}

function micAlive() {
  const t = stream && stream.getAudioTracks && stream.getAudioTracks()[0];
  return !!(t && t.readyState === "live");
}

async function openMic() {           // set up the audio graph once; streams while capturing
  if (SPEAKER_ONLY) return false;    // this page is a loudspeaker, not a microphone
  // "if (source) return true" handed back whatever graph existed last time, including one
  // whose track had ENDED -- iOS ends the capture track on its own when the audio session is
  // reconfigured, which is exactly what pairing a Bluetooth speaker does. The graph then
  // looked fine, onaudioprocess never fired again, and the phone reported "mic is open but
  // hearing nothing". Check the track is actually live, and rebuild from scratch if it is not.
  if (source && micAlive()) return true;
  if (source) {
    report("mic-rebuild", "capture track was not live; rebuilding the graph");
    try { closeMic(); } catch (e) {}
    source = null; processor = null;
  }
  try {
    // NO echoCancellation, NO noiseSuppression. Both put iOS into its voice-processing audio
    // path, and once the DM has spoken through the phone's own speaker that path gates the
    // microphone almost shut: a hands-free session recorded 7.4 seconds of audio across 68
    // seconds of listening, peaking at 0.013 -- while push-to-talk minutes earlier, before the
    // DM had made a sound, peaked at 0.98. Echo is handled on the server, where it can be
    // judged against what the DM actually said.
    // RAW MICROPHONE. No echo cancellation, no noise suppression, no automatic gain -- every
    // one of these is a browser "service" that on iOS drags the capture into the voice-
    // processing audio path, and once the phone's own speaker is live that path clamps the
    // microphone almost shut. Measured: 7.4 seconds of audio across 68 seconds of listening,
    // peaking at 0.013, against 0.98 for the same phone with the speaker idle.
    //
    // Nothing is lost by refusing them. The server's VAD learns the room's noise floor from
    // the stream and moves its threshold with it, and echo is judged there too -- against what
    // the DM actually said, and only while it is speaking.
    const want = {
      echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      // some builds only honour the vendor-prefixed forms; ask both ways, request nothing extra
      googEchoCancellation: false, googNoiseSuppression: false, googAutoGainControl: false,
    };
    stream = await navigator.mediaDevices.getUserMedia({ audio: want });
    // WHICH MICROPHONE DECIDES WHERE THE SOUND COMES OUT. That is the whole problem, and
    // pinning the phone's own microphone was making it worse, not better.
    //
    // While a capture is open, iOS is in PlayAndRecord, and A2DP is not offered in that
    // category. So there are exactly two coherent arrangements:
    //
    //   Bluetooth mic  -> the session is on HFP, and output goes to the BLUETOOTH SPEAKER.
    //                     The microphone is 8kHz and further away. The room hears the DM.
    //   built-in mic   -> no Bluetooth route is available at all, and output falls back to
    //                     the PHONE'S speaker. Better microphone, worse for a table.
    //
    // Forcing the built-in one therefore silently chose "no Bluetooth output", which is
    // exactly what was reported. So: detect what is actually there and pick the arrangement
    // that works for the room, preferring Bluetooth when a Bluetooth input exists, and falling
    // back to the phone when it does not. MIC_PREF can override either way.
    // Dump what iOS is actually offering, ALWAYS, before anything can throw. The previous
    // version reported the choice from inside the try, so when enumerateDevices() failed the
    // whole diagnostic vanished and the log showed nothing at all -- which is exactly what
    // happened, and left the one question that matters unanswerable from this end: does the
    // Bluetooth speaker present itself as an input device, or only as an output?
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      report("mic-devices", all.filter((d) => d.kind === "audioinput")
        .map((d) => `${d.label || "(unlabelled)"}${d.deviceId ? "" : " [no id]"}`)
        .join(" | ") || "(none listed)");
    } catch (e) {
      report("mic-devices", `enumerateDevices FAILED: ${e && e.name} ${e && e.message}`);
    }
    try {
      const ins = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "audioinput" && d.deviceId);
      // Anything that is not the phone itself. Prefer a device whose name says outright
      // what it is, then fall back to any input that simply is not built in -- which is
      // what catches a speakerphone that only ever offers a brand name.
      const bt = ins.find((d) => NOT_BUILT_IN_RE.test(d.label))
              || ins.find((d) => isExternalMic(d.label));
      const built = ins.find((d) => BUILT_IN_RE.test(d.label))
                 || ins.find((d) => !isExternalMic(d.label));
      const pref = MIC_PREF === "phone" ? built
                 : MIC_PREF === "bluetooth" ? bt
                 : (bt || built);              // auto: Bluetooth if it is there, else the phone
      const cur = stream.getAudioTracks()[0];
      const curId = cur && cur.getSettings ? cur.getSettings().deviceId : null;
      report("mic-choice", `mode=${MIC_PREF} bt=${bt ? bt.label.slice(0, 28) : "none"} `
        + `built=${built ? built.label.slice(0, 28) : "none"} `
        + `-> ${pref ? pref.label.slice(0, 28) : "whatever iOS gave"}`
        + (bt ? "  (bluetooth input present: output can reach the speaker)"
              : "  (no bluetooth input: the DM will come out of the phone)"));
      if (pref && curId && pref.deviceId !== curId) {
        stream.getTracks().forEach((t) => t.stop());
        stream = await navigator.mediaDevices.getUserMedia({
          audio: Object.assign({}, want, { deviceId: { exact: pref.deviceId } }) });
      }
      // A microphone that is not the phone's own belongs to something with a speaker in it --
      // a speakerphone, a headset, a conference puck. That device carries both directions at
      // once, which is the thing half-duplex existed to fake, so switch it off and give the
      // table back barge-in. Announced rather than silent: "why can I interrupt tonight?"
      // should have an answer in the log.
      const chosen = (stream.getAudioTracks()[0] || {}).label || (pref && pref.label) || "";
      if (HALF_DUPLEX && isExternalMic(chosen)) {
        HALF_DUPLEX = false;
        cancelMicReopen(); micHeld = false;
        report("duplex", `full duplex: '${chosen.slice(0, 32)}' has a microphone of its own, `
          + "so it can play and listen at the same time");
      }
    } catch (e) {
      // Say what went wrong. A silent catch here is why the device diagnostic never appeared.
      report("mic-choice-failed", `${e && e.name}: ${e && e.message}`);
      // An unsatisfiable deviceId throws AFTER the old stream has been stopped, i.e. no
      // microphone at all. Losing the choice is survivable; losing the mic is not.
      if (!micAlive()) {
        try { stream = await navigator.mediaDevices.getUserMedia({ audio: want }); } catch (e2) {}
      }
    }
  } catch (err) { addMsg("sys", "mic blocked — needs HTTPS (open the https tunnel URL)"); return false; }
  ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
  try { await ctx.resume(); } catch (e) { report("mic-ctx", `resume failed: ${e && e.name}`); }
  source = ctx.createMediaStreamSource(stream);
  processor = ctx.createScriptProcessor(4096, 1, 1);
  // A ScriptProcessor only runs while something pulls it, but pulling it through
  // ctx.destination keeps the OUTPUT device engaged, which is the other half of what puts iOS
  // into that same voice-processing mode. A MediaStreamDestination pulls it just as well and
  // touches no speaker.
  const sink = ctx.createMediaStreamDestination
    ? ctx.createMediaStreamDestination()
    : (() => { const g = ctx.createGain(); g.gain.value = 0; g.connect(ctx.destination); return g; })();
  processor.onaudioprocess = (e) => {
    // An exception in here dies silently and the mic simply stops existing, which is
    // indistinguishable from "the DM is ignoring me". Count what actually leaves the phone.
    micFrames++;
    if (!capturing) { micSkips.notCapturing++; return; }
    if (ws.readyState !== 1) { micSkips.socket++; return; }
    // Always uploading, even while the DM speaks -- that's what makes it possible to answer
    // the instant it stops, or talk over it. Echo is handled server-side (stricter threshold
    // while the DM has the floor, plus a transcript check before it's cut off).
    const ch = e.inputBuffer.getChannelData(0);
    // Track the loudest thing we have heard, so "it can't hear me" has a number attached to it
    // instead of a guess.
    let pk = 0;
    for (let i = 0; i < ch.length; i += 16) { const a = ch[i] < 0 ? -ch[i] : ch[i]; if (a > pk) pk = a; }
    if (pk > micPeak) micPeak = pk;
    micLevel = micLevel * 0.9 + pk * 0.1;
    const pcm = toInt16(downsample(ch, ctx.sampleRate));
    ws.send(UPLINK_ULAW ? toUlaw(pcm).buffer : pcm.buffer);
    micSent++;
  };
  source.connect(processor); processor.connect(sink);
  return true;
}
function closeMic() {
  if (processor) processor.disconnect();
  if (source) { source.disconnect(); source = null; }
  if (stream) stream.getTracks().forEach((t) => t.stop());
}

// -- push-to-talk --
async function startRec() {
  // In half-duplex the microphone is deliberately shut while the DM talks, so hands-free
  // cannot hear a barge-in -- there is nothing listening to barge into. Hold-to-speak IS the
  // interrupt: pressing it is an explicit choice to open the mic, and iOS moving the audio
  // back to the phone for those few seconds is a consequence the player just asked for.
  if (HALF_DUPLEX && handsfree) {
    if (speechPlaying) interruptDM();          // cut the narration off first
    cancelMicReopen();
    unlockAudio();
    try { if (ctx && ctx.state === "suspended") await ctx.resume(); } catch (e) {}
    if (!(await openMic())) { report("duplex", "hold-to-speak could not open the mic"); return; }
    micHeld = false;                             // the player asked for it back; it is theirs
    capturing = recording = true;
    talk.classList.add("rec"); talk.textContent = "Listening…";
    ws.send(JSON.stringify({ type: "start" }));
    return;
  }
  if (recording || handsfree || !ws || ws.readyState !== 1) return;
  unlockAudio();
  if (!(await openMic())) return;
  capturing = recording = true; talk.classList.add("rec"); talk.textContent = "Listening…";
  ws.send(JSON.stringify({ type: "start" }));
}
function stopRec() {
  if (!recording) return;
  capturing = recording = false; talk.classList.remove("rec"); talk.textContent = "Hold to Speak";
  ws.send(JSON.stringify({ type: "end" }));
  closeMic();
  // hand the mic back to hands-free rather than leaving it dead until the next DM line
  if (HALF_DUPLEX && handsfree) scheduleMicReopen(300);
}
for (const [down, up] of [["touchstart", "touchend"], ["mousedown", "mouseup"]]) {
  talk.addEventListener(down, (e) => { e.preventDefault(); startRec(); }, { passive: false });
  talk.addEventListener(up, (e) => { e.preventDefault(); stopRec(); }, { passive: false });
}
talk.addEventListener("touchcancel", stopRec);

// -- hands-free (server VAD auto-detects your turns) --
async function toggleHF() {
  handsfree = !handsfree;
  if (handsfree) {
    unlockAudio();
    if (!(await openMic())) { handsfree = false; return; }
    capturing = true; hf.classList.add("on");
    turnOver = true; setHfState("listening");
    talk.disabled = !!HALF_DUPLEX ? false : true;   // half-duplex: it is the barge-in
    if (HALF_DUPLEX) talk.textContent = "Hold to interrupt / speak";
    ws.send(JSON.stringify({ type: "listen" }));
    micFrames = micSent = 0; micSkips.notCapturing = micSkips.socket = 0;
    const tr = stream && stream.getAudioTracks()[0];
    // What we ASKED for and what iOS granted are different things, and the gap is invisible
    // unless it is printed: constraints it dislikes are ignored in silence.
    let got = "";
    try {
      const st = tr && tr.getSettings ? tr.getSettings() : {};
      got = ` ec=${st.echoCancellation} ns=${st.noiseSuppression} agc=${st.autoGainControl}`;
    } catch (e) {}
    report("mic-on", `ctx=${ctx && ctx.state} sr=${ctx && ctx.sampleRate} `
      + `track=${tr ? tr.readyState : "none"} muted=${tr && tr.muted} `
      + `enabled=${tr && tr.enabled}${got} label=${tr ? tr.label.slice(0, 40) : "-"}`);
    // The graph can be built correctly and still never fire a callback -- that is the exact
    // failure that read as "it can't hear me", with nothing anywhere to say so.
    micPeak = micLevel = 0;
    // Keep reporting, not once. A mic that is open but gated reports perfectly healthy at
    // two seconds and then delivers silence for the next minute, which is exactly what
    // happened -- and nothing anywhere said so.
    let checks = 0;
    clearInterval(micTimer);
    micTimer = setInterval(() => {
      if (!handsfree) { clearInterval(micTimer); return; }
      checks++;
      const dead = micFrames === 0 ? "  <-- onaudioprocess NEVER FIRED"
        : "";
      report("mic-check", `frames=${micFrames} sent=${micSent} `
        + `peak=${micPeak.toFixed(3)} level=${micLevel.toFixed(3)} `
        + `skipped(notCapturing=${micSkips.notCapturing} socket=${micSkips.socket}) `
        + `ctx=${ctx && ctx.state}${dead}`);
      // iOS ends the capture track whenever it reshuffles the audio session -- a Bluetooth
      // speaker connecting, a call, another app taking audio. Rebuild rather than sitting
      // there open and deaf until somebody notices.
      const tr0 = stream && stream.getAudioTracks && stream.getAudioTracks()[0];
      // Only a problem when we ASKED for the phone. On auto or bluetooth, a Bluetooth
      // input is the arrangement we want, not a theft.
      const stolen = MIC_PREF === "phone" && tr0 && isExternalMic(tr0.label || "");
      // NOT while half-duplex has deliberately shut the microphone. This watchdog exists to
      // revive a mic that iOS killed, and it cannot tell that from one we closed on purpose --
      // so every 2.5 seconds it "rescued" the mic in the middle of the DM speaking, which put
      // iOS straight back into PlayAndRecord and moved the audio off the Bluetooth speaker
      // mid-sentence. That is exactly the "starts on the speaker then swaps to the phone".
      if (HALF_DUPLEX && micHeld) {
        micPeak = 0; micFrames = 0;
        // Held is normal WHILE the DM talks. Held long after it stopped means the handover
        // never completed, and sitting here quietly is how the phone ends up deaf for the rest
        // of the session -- so push it rather than wait to be asked.
        if (!speechPlaying && !speechQ.length && _quietSince && Date.now() - _quietSince > 4000
            && document.visibilityState === "visible" && !_reopenTimer) {
          releaseMic("watchdog");
        }
        return;
      }
      if (!micAlive() || micFrames === 0 || stolen) {
        report("mic-dead", `alive=${micAlive()} frames=${micFrames} `
          + `stolen=${stolen ? (tr0.label || "").slice(0, 30) : "no"} -- rebuilding`);
        closeMic(); source = null; processor = null;
        openMic().then((ok) => { if (ok && handsfree) capturing = true; });
      }
      // NO "open but hearing nothing" warning. It fired on a quiet ROOM, which is most of a
      // session: a speakerphone runs its own noise gate and delivers near-zero between
      // utterances, and this check could not tell that from a dead capture. On a working
      // PowerConf it warned while the table was being heard perfectly. A warning that cries
      // wolf during normal play is worse than none -- it teaches the table to distrust the
      // app while it is working. A genuinely dead microphone is still caught above, by the
      // track ending or frames never arriving, both of which are facts rather than guesses.
      micFrames = 0;
      micPeak = 0;
    }, 2500);
  } else {
    clearInterval(micTimer); micTimer = 0;
    cancelMicReopen(); micHeld = false;
    capturing = false; hf.classList.remove("on", "busy"); hf.textContent = "🎙️ Auto";
    talk.disabled = false; ws.send(JSON.stringify({ type: "stop_listen" })); closeMic();
  }
}
hf.addEventListener("click", toggleHF);
if (SPEAKER_ONLY) {
  // No capture UI: one accidental tap re-creates the very conflict this exists to avoid.
  // But iOS will not play a sound until the page has been TOUCHED, so there has to be exactly
  // one thing to touch -- hide everything and this page connects and then sits there silent.
  for (const id of ["handsfree", "enroll"]) { const b = el(id); if (b) b.hidden = true; }
  const go = el("talk");
  if (go) {
    go.textContent = "🔊 Tap to start the speaker";
    go.disabled = false;
    const arm = (e) => {
      if (e) e.preventDefault();
      unlockAudio();
      go.textContent = "🔊 Speaker ready";
      go.disabled = true;
      addMsg("sys", "Sound is on. Route this device to the Bluetooth speaker.");
    };
    // replaceWith(cloneNode) strips the push-to-talk listeners bound earlier
    const fresh = go.cloneNode(true);
    go.parentNode.replaceChild(fresh, go);
    fresh.addEventListener("click", arm);
    fresh.addEventListener("touchend", arm, { passive: false });
  }
  addMsg("sys", "🔊 Speaker mode — plays the DM out loud, never opens a microphone. "
               + "Because it never records, iOS will send this to Bluetooth.");
} else if (MIC_ONLY) {
  addMsg("sys", "🎙️ Microphone mode — this device listens; the DM plays on the speaker.");
}
if (HALF_DUPLEX) {
  report("duplex", "half — mic closes while the DM speaks so Bluetooth can have the audio");
  addMsg("sys", "🔉 Half-duplex — the mic closes while the DM talks, so a Bluetooth speaker "
              + "gets the audio. You cannot interrupt mid-sentence in this mode.");
}
el("stop").addEventListener("click", interruptDM);

// enroll a player's voice via an inline form (no prompt() — iOS blocks those)
//
// Tap your CHARACTER, then type your own name. It used to ask for both as free text, which
// meant every player at the table spelling "Gus of The Slippery Rock" into a phone keyboard
// before the session could start — and getting it wrong made a profile nobody matched.
const enrollPanel = document.getElementById("enrollPanel");
const epWho = document.getElementById("epWho");
const eRec = document.getElementById("eRec");
let epChar = "";

function setEnrollChar(name, btn) {
  epChar = name;
  for (const b of epWho.querySelectorAll("button")) b.classList.toggle("on", b === btn);
  eRec.disabled = false;
  eRec.textContent = "Hold to record your voice";
}

async function loadEnrollParty() {
  epWho.textContent = "";
  epChar = ""; eRec.disabled = true; eRec.textContent = "Pick your character first";
  const name = current || "";
  try {
    const r = await fetch("/party/" + encodeURIComponent(name));
    const { party } = await r.json();
    if (!party || !party.length) { epWho.textContent = "no characters in this campaign yet"; return; }
    for (const p of party) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "epchip"; b.textContent = p.name;
      b.addEventListener("click", () => setEnrollChar(p.name, b));
      epWho.appendChild(b);
    }
  } catch (e) { epWho.textContent = "couldn't load the party"; }
}

document.getElementById("enroll").addEventListener("click", () => {
  enrollPanel.hidden = !enrollPanel.hidden;
  if (!enrollPanel.hidden) loadEnrollParty();
});
async function startEnroll() {
  if (recording || !ws || ws.readyState !== 1) return;
  if (!epChar) { addMsg("sys", "tap your character first"); return; }
  const player = (document.getElementById("epName").value || "").trim();
  if (!player) { addMsg("sys", "type your name first"); return; }
  unlockAudio();
  ws.send(JSON.stringify({ type: "enroll", player, character: epChar }));
  if (!(await openMic())) return;
  capturing = recording = true; eRec.classList.add("rec"); eRec.textContent = "Recording… release when done";
  ws.send(JSON.stringify({ type: "start" }));
}
function stopEnroll() {
  if (!recording) return;
  capturing = recording = false; eRec.classList.remove("rec"); eRec.textContent = "Hold to record your voice";
  ws.send(JSON.stringify({ type: "end" }));
  closeMic();
}
for (const [d, u] of [["touchstart", "touchend"], ["mousedown", "mouseup"]]) {
  eRec.addEventListener(d, (e) => { e.preventDefault(); startEnroll(); }, { passive: false });
  eRec.addEventListener(u, (e) => { e.preventDefault(); stopEnroll(); }, { passive: false });
}
eRec.addEventListener("touchcancel", stopEnroll);

// ---- campaigns: pick, inspect, create, clear ----
const panel = el("campaignPanel"), cpBody = el("cpBody");
let current = localStorage.getItem("dm.campaign") || "", openDetail = null;

const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function loadCampaigns() {
  try {
    const d = await api("/campaigns");
    // Once you've picked one, that's the one you're playing. Taking the server's idea of
    // "current" back here is what left the badge on the old campaign after switching.
    if (!current) current = d.current || "";
    if (current) localStorage.setItem("dm.campaign", current);
    el("campaignName").textContent = current || "Campaigns";
    render(d.campaigns || []);
  } catch (e) { cpBody.innerHTML = `<div class="cp-empty">Couldn't load campaigns.</div>`; }
}

function render(list) {
  if (!list.length) {
    cpBody.innerHTML = `<div class="cp-empty">No campaigns yet — name one below to begin.</div>`;
    return;
  }
  cpBody.innerHTML = list.map((c) => {
    const party = (c.party || []).filter(Boolean).join(", ");
    const on = c.name === current;
    return `<div class="cp-card${on ? " on" : ""}">
      <div class="cp-top">
        <div>
          <div class="cp-name">${esc(c.name)}${on ? ' <span class="cp-now">playing</span>' : ""}</div>
          <div class="cp-meta">${c.session ? `${c.session} session${c.session === 1 ? "" : "s"}` : "not started"}${party ? " · " + esc(party) : ""}</div>
          ${c.premise ? `<div class="cp-premise">${esc(c.premise)}</div>` : ""}
        </div>
      </div>
      <div class="cp-actions">
        ${on ? "" : `<button class="hf tiny" data-play="${esc(c.name)}">▶ Play</button>`}
        <button class="hf tiny" data-detail="${esc(c.name)}">Sessions</button>
        <button class="hf tiny" data-newsession="${esc(c.name)}">New session</button>
        <button class="hf tiny warn" data-reset="${esc(c.name)}">Clear history</button>
        <button class="hf tiny danger" data-del="${esc(c.name)}">Delete</button>
      </div>
      ${openDetail === c.name ? `<div class="cp-detail" id="d-${esc(c.name)}">loading…</div>` : ""}
    </div>`;
  }).join("");
  if (openDetail) fillDetail(openDetail);
}

async function fillDetail(name) {
  const box = document.getElementById("d-" + name);
  if (!box) return;
  try {
    const d = await api("/campaigns/" + encodeURIComponent(name));
    const party = (d.party || []).map((p) => `${esc(p.character)} <i>(${esc(p.player)})</i>`).join(", ");
    box.innerHTML = `
      ${party ? `<div><b>Party</b> ${party}</div>` : ""}
      ${d.location ? `<div><b>Where</b> ${esc(d.location)}</div>` : ""}
      ${d.recap ? `<div><b>Last time</b> ${esc(d.recap)}</div>` : ""}
      ${(d.quests || []).length ? `<div><b>Threads</b> ${d.quests.map(esc).join("; ")}</div>` : ""}
      <div><b>Sessions</b> ${d.session || 0} · <b>world notes</b> ${d.notes || 0} lines</div>
      ${(d.sessions || []).length
        ? `<ul class="cp-sessions">${d.sessions.map((s) => `<li>${esc(s.replace(/^=== | ===$/g, ""))}</li>`).join("")}</ul>`
        : `<div class="cp-empty">No sessions played yet.</div>`}`;
  } catch (e) { box.textContent = "Couldn't load details."; }
}

// destructive actions need a second tap rather than a blocking dialog
function arm(btn, label, run) {
  if (btn.dataset.armed) { run(); return; }
  const was = btn.textContent;
  btn.dataset.armed = "1"; btn.textContent = label;
  setTimeout(() => { delete btn.dataset.armed; btn.textContent = was; }, 4000);
}

cpBody.addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  const { play, detail, reset, del, newsession } = b.dataset;
  try {
    if (play) { switchTo(play); }
    else if (detail) { openDetail = openDetail === detail ? null : detail; await loadCampaigns(); }
    else if (newsession) arm(b, "Start a new session?", async () => {
      await api(`/campaigns/${encodeURIComponent(newsession)}/new-session`, { method: "POST" });
      addMsg("sys", `next time you begin, ${newsession} opens a new session`);
      if (newsession === current) switchTo(current); else loadCampaigns();
    });
    else if (reset) arm(b, "Clear history? (keeps party, arc, safety)", async () => {
      await api(`/campaigns/${encodeURIComponent(reset)}/reset`, { method: "POST" });
      addMsg("sys", `cleared history for ${reset}`);
      if (reset === current) switchTo(current); else loadCampaigns();
    });
    else if (del) arm(b, "Delete for good?", async () => {
      await api("/campaigns/" + encodeURIComponent(del), { method: "DELETE" });
      addMsg("sys", `deleted ${del}`);
      if (del === current) current = "";
      loadCampaigns();
    });
  } catch (err) { addMsg("sys", "⚠ " + err.message); }
});

function switchTo(name) {
  current = name;
  el("campaignName").textContent = name;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "campaign", name }));
  log.innerHTML = ""; addMsg("sys", "▶ " + name); refreshStart();
  panel.hidden = true;
  loadCampaigns();
}

let startNudged = false;
// ---- start of session ----
// Opening a campaign and resuming one are different asks, and neither should depend on the
// player guessing the right words to say into a mic.
async function refreshStart() {
  const bar = el("startBar");
  if (!current) { bar.hidden = true; return; }
  try {
    const d = await api("/campaigns/" + encodeURIComponent(current));
    // the server decides this; deriving it here separately is how the button came to say
    // "Begin the campaign" while the server asked the DM to recap a session that never was
    const fresh = d.unplayed;
    el("startBtn").textContent = fresh ? "▶ Begin the campaign" : "▶ Recap and continue";
    // an unplayed campaign is waiting on one tap; say so instead of sitting there silently
    if (fresh && !startNudged) { startNudged = true; addMsg("sys", "tap ▶ Begin the campaign to start"); }
    el("startHint").textContent = fresh
      ? `${esc(current)} — nothing played yet`
      : `${esc(current)} — session ${d.session}, ${esc((d.scene || "").slice(0, 60))}`;
    bar.hidden = false;
  } catch (e) { bar.hidden = true; }
}
el("startBtn").addEventListener("click", () => {
  if (!ws || ws.readyState !== 1) return;
  unlockAudio();                       // this tap is what lets iOS play audio at all
  holdWake();                          // and it is the only moment iOS grants a wake lock
  el("startBar").hidden = true;
  addMsg("sys", "▶ starting…");
  ws.send(JSON.stringify({ type: "begin" }));
});

// ---- initiative ----
// Combat is unreadable by voice alone — who's up, who's next, what round. This is the one
// thing worth putting permanently on screen during a fight.
// ---- who is at this table, on which phone -------------------------------------------
// Each player taps their own character once. That is what makes "who said that" a fact
// rather than a voice-similarity score -- at a real seven-player session the enrolled
// profiles sat up to 0.703 similar to one another and correctly refused to name anybody.
let ROSTER = [], MY_SEAT = "", SPEAKER_ON = true, MY_SID = "";
// what the server last told us about the table, and whether we have already
// re-claimed on this connection
let LAST_SEATS = [], _reclaimed = false;

function showTable(m) {
  LAST_SEATS = m.seats || [];
  const mine = (m.seats || []).find((s) => s.sid && s.sid === MY_SID);
  if (mine) { MY_SEAT = mine.character || ""; SPEAKER_ON = !!mine.speaker; }
  const el0 = el("seatName");
  if (el0) el0.textContent = MY_SEAT || "Pick character";
  const sp = el("spkBtn");
  if (sp) {
    sp.textContent = SPEAKER_ON ? "🔊 Speaker" : "🔇 Muted";
    sp.classList.toggle("on", SPEAKER_ON);
  }
  // Nobody holding the audio is a table that hears nothing, so it is worth saying out loud
  // on screen rather than leaving them to work it out.
  if ((m.speakers || 0) === 0) addMsg("sys", "no device is playing the DM — tap 🔊 on one");
}

function pickSeat() {
  const box = el("seatPanel"), list = el("seatList");
  if (!box || !list) return;
  list.innerHTML = ROSTER.map((c) => `<button class="seatpick" data-c="${c}">${c}</button>`).join("")
    + `<button class="seatpick" data-c="">(nobody — just watching)</button>`;
  list.querySelectorAll(".seatpick").forEach((b) => b.addEventListener("click", () => {
    const c = b.dataset.c || "";
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "seat", character: c }));
    MY_SEAT = c;
    const n = el("seatName"); if (n) n.textContent = c || "Pick character";
    // remembered per device, so a reload does not make everyone pick again mid-session
    try { localStorage.setItem("dm.seat", c); } catch (e) {}
    box.hidden = true;
  }));
  box.hidden = !box.hidden;
}

function toggleSpeaker() {
  SPEAKER_ON = !SPEAKER_ON;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "audio", on: SPEAKER_ON }));
  const sp = el("spkBtn");
  if (sp) { sp.textContent = SPEAKER_ON ? "🔊 Speaker" : "🔇 Muted"; sp.classList.toggle("on", SPEAKER_ON); }
}

// ---- ending your own turn -------------------------------------------------------------
// A player's turn ends when they say so, which is right and is also why a real session never
// left round 1: nobody knew the phrase and the DM never asked. Now the server says whose turn
// it is out loud AND the player whose turn it is gets a button, because a tap beats a phrase
// when seven people are talking.
function showEndTurn(who) {
  const b = el("endTurn");
  if (!b) return;
  const mine = !!who && !!MY_SEAT && who.toLowerCase() === MY_SEAT.toLowerCase();
  b.hidden = !mine;
  if (mine) b.textContent = "✔ End my turn";
}

function endMyTurn() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "text", text: "That's the end of my turn." }));
  const b = el("endTurn"); if (b) b.hidden = true;
}

function showInitiative(c) {
  const bar = el("initBar");
  if (!c.in_combat) { bar.hidden = true; bar.innerHTML = ""; return; }
  if (c.pending && c.pending.length) {
    bar.innerHTML = `<div class="ini-wait">⚔ Roll initiative — waiting on
      <b>${c.pending.map(esc).join(", ")}</b></div>`;
    bar.hidden = false;
    return;
  }
  const cells = (c.order || []).map((o) => {
    const now = o.name === c.turn;
    // How much the party has put into this one. The DM never reads hit points aloud -- that
    // ruins the guesswork -- but a bar is not the same as saying the number, and without it
    // nobody can judge whether to swing again or fall back.
    const hurt = (typeof o.hurt === "number")
      ? `<span class="ini-hp" title="${o.dealt} damage dealt">
           <span class="ini-hp-fill" style="width:${Math.max(0, Math.min(100, o.hurt))}%"></span>
         </span>` : "";
    return `<span class="ini${now ? " now" : ""}${o.pc ? " pc" : ""}">
      <b>${o.init ?? "?"}</b> ${esc(o.name)}${hurt}
      <i>${esc(o.state || "")}</i></span>`;
  }).join("");
  bar.innerHTML = `<div class="ini-round">Round ${c.round}</div><div class="ini-row">${cells}</div>`;
  bar.hidden = false;
}

// ---- battle map ----
// Arrives mid-session, seconds after the DM calls for it. Shown but never forced over the
// screen mid-turn: the button appears and the panel opens on the first map of a scene only.
const mapPanel = el("mapPanel"), mapBody = el("mapBody");
let lastMap = null;

function showMap(title, svg, open) {
  lastMap = { title, svg };
  el("mapTitle").textContent = title || "Battle map";
  mapBody.innerHTML = svg;
  el("mapBtn").hidden = false;
  if (open) mapPanel.hidden = false;
}
el("mapBtn").addEventListener("click", () => { mapPanel.hidden = !mapPanel.hidden; });
el("mapClose").addEventListener("click", () => { mapPanel.hidden = true; });

// ---- saved maps: every place ever drawn, kept until cleared ----
const mapLibList = el("mapLibList");
async function loadMapLib() {
  const camp = current || "default";
  let saved = [];
  try { saved = (await api("/maps/" + encodeURIComponent(camp))).maps || []; } catch (e) {}
  mapLibList.innerHTML = "";
  if (!saved.length) { mapLibList.innerHTML = "<div class='msg sys'>no maps saved yet</div>"; return; }
  saved.forEach((m) => {
    const row = document.createElement("div");
    row.className = "librow";
    const open = document.createElement("button");
    open.className = "hf tiny";
    open.textContent = "🗺 " + m.title;
    open.title = m.rooms + " areas · " + m.seen + " explored";
    open.onclick = () => window.open("/maps/" + encodeURIComponent(camp) + "/" + m.slug, "_blank");
    const del = document.createElement("button");
    del.className = "hf tiny";
    del.textContent = "✕";
    del.onclick = async () => {
      await fetch("/maps/" + encodeURIComponent(camp) + "?slug=" + encodeURIComponent(m.slug),
                  { method: "DELETE" });
      loadMapLib();
    };
    row.append(open, del);
    mapLibList.appendChild(row);
  });
  const all = document.createElement("button");
  all.className = "hf tiny";
  all.textContent = "🗑 Clear all maps";
  all.onclick = async () => {
    if (!confirm("Delete every saved map for this campaign?")) return;
    await fetch("/maps/" + encodeURIComponent(camp), { method: "DELETE" });
    loadMapLib();
  };
  mapLibList.appendChild(all);
}
el("mapLib").addEventListener("click", () => {
  mapLibList.hidden = !mapLibList.hidden;
  if (!mapLibList.hidden) loadMapLib();
});
el("mapPrint").addEventListener("click", () => {
  if (!lastMap) return;
  const w = window.open("", "_blank");
  if (!w) { addMsg("sys", "allow pop-ups to print the map"); return; }
  w.document.write(`<!doctype html><title>${lastMap.title}</title>` +
    `<style>body{margin:0;display:flex;align-items:center;justify-content:center}` +
    `svg{max-width:100%;height:auto}@media print{body{margin:0}}</style>${lastMap.svg}`);
  w.document.close(); w.focus(); setTimeout(() => w.print(), 350);
});

// ---- character sheets ----
// The sheet itself is server-rendered HTML so the same page prints and displays; shown here
// in an iframe rather than a new tab, which mobile browsers like to block.
const sheetPanel = el("sheetPanel"), shBody = el("shBody");

async function loadSheets() {
  if (!current) { shBody.innerHTML = `<div class="cp-empty">Pick a campaign first.</div>`; return; }
  el("shPrint").href = `/sheet/${encodeURIComponent(current)}`;
  try {
    const d = await api("/party/" + encodeURIComponent(current));
    if (!(d.party || []).length) {
      shBody.innerHTML = `<div class="cp-empty">No characters in ${esc(current)} yet.<br>
        Say “import my character” with your D&amp;D Beyond id, or use ＋ Voice to add a player.</div>`;
      return;
    }
    shBody.innerHTML = d.party.map((p) => `
      <div class="cp-card">
        <div class="cp-name">${esc(p.name)}</div>
        <div class="cp-meta">${esc((p.notes || "").split("(")[0].trim())} · AC ${p.ac} · ${p.hp}/${p.max_hp} HP</div>
        <div class="cp-actions">
          <button class="hf tiny" data-sheet="${esc(p.name)}">View sheet</button>
          <a class="hf tiny" href="/sheet/${encodeURIComponent(current)}?who=${encodeURIComponent(p.name)}"
             target="_blank" rel="noopener">🖨 Print</a>
        </div>
        <div class="sh-frame" id="sh-${esc(p.name)}" hidden></div>
      </div>`).join("");
  } catch (e) { shBody.innerHTML = `<div class="cp-empty">Couldn't load the party.</div>`; }
}

shBody.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-sheet]"); if (!b) return;
  const box = document.getElementById("sh-" + b.dataset.sheet);
  if (!box) return;
  if (box.hidden) {
    if (!box.firstChild) {
      const f = document.createElement("iframe");
      f.src = `/sheet/${encodeURIComponent(current)}?who=${encodeURIComponent(b.dataset.sheet)}`;
      f.addEventListener("load", () => {
        try { f.style.height = f.contentDocument.documentElement.scrollHeight + "px"; }
        catch (e) {}
      });
      box.appendChild(f);
    }
    box.hidden = false; b.textContent = "Hide sheet";
  } else { box.hidden = true; b.textContent = "View sheet"; }
});

el("sheetBtn").addEventListener("click", () => {
  sheetPanel.hidden = !sheetPanel.hidden;
  if (!sheetPanel.hidden) loadSheets();
});
el("shClose").addEventListener("click", () => { sheetPanel.hidden = true; });

// ---- feedback: captured mid-play, reviewed here ----
const notesPanel = el("notesPanel"), fbBody = el("fbBody");

async function loadNotes() {
  try {
    const d = await api("/feedback");
    const open = (d.items || []).filter((i) => i.status !== "done");
    el("notesCount").textContent = open.length ? " " + open.length : "";
    fbBody.innerHTML = (d.items || []).length
      ? d.items.map((i) => `<div class="cp-card${i.status === "done" ? " fb-done" : ""}">
          <div class="fb-text">${esc(i.text)}</div>
          <div class="cp-meta">${esc(i.ts)}${i.campaign ? " · " + esc(i.campaign) : ""}${i.session ? " · session " + i.session : ""}</div>
          <div class="cp-actions">
            <button class="hf tiny" data-fb="${i.id}" data-to="${i.status === "done" ? "open" : "done"}">
              ${i.status === "done" ? "↩ Reopen" : "✓ Done"}</button>
            <button class="hf tiny danger" data-fb="${i.id}" data-to="deleted">Delete</button>
          </div></div>`).join("")
      : `<div class="cp-empty">No feedback yet. Say “feedback …” during play, or add one below.</div>`;
  } catch (e) { fbBody.innerHTML = `<div class="cp-empty">Couldn't load feedback.</div>`; }
}

fbBody.addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-fb]"); if (!b) return;
  const run = async () => {
    await api(`/feedback/${b.dataset.fb}/${b.dataset.to}`, { method: "POST" });
    loadNotes();
  };
  try { b.dataset.to === "deleted" ? arm(b, "Delete?", run) : await run(); }
  catch (err) { addMsg("sys", "⚠ " + err.message); }
});

el("notesBtn").addEventListener("click", () => {
  notesPanel.hidden = !notesPanel.hidden;
  if (!notesPanel.hidden) loadNotes();
});
el("fbClose").addEventListener("click", () => { notesPanel.hidden = true; });
el("fbAdd").addEventListener("click", async () => {
  const text = el("fbText").value.trim();
  if (!text) return;
  try {
    await api("/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
                             body: JSON.stringify({ text, campaign: current }) });
    el("fbText").value = ""; loadNotes();
  } catch (err) { addMsg("sys", "⚠ " + err.message); }
});

// ---- pick up a new build without interrupting play ----
// Only ever reloads while genuinely idle: not mid-turn, not speaking, not hands-free-armed,
// and with no panel open. Otherwise it waits for the next quiet moment.
// ---- coming back from a locked phone / backgrounded tab ----
// iOS suspends the AudioContext and drops the socket when the app isn't in front, and nothing
// recovers on its own: you come back to a page that looks fine but can't hear you. Rebuild
// every piece that the OS is allowed to tear down.
async function wakeUp() {
  if (document.visibilityState !== "visible") return;
  // NOT while half-duplex is deliberately holding the context down for playback -- coming
  // back to the page mid-narration would resume it and take the audio off the speaker.
  if (ctx && ctx.state === "suspended" && !micHeld) {
    try { await ctx.resume(); } catch (e) {}
  }
  if (_wakeWanted && !_wakeLock) holdWake();       // a lock is dropped whenever we go hidden
  if (!ws || ws.readyState > 1) { reconnects = Math.max(reconnects, 1); connect(); return; }
  // Coming back to the front is the FIRST moment a held microphone can actually be reopened --
  // iOS will not give a hidden page a capture device, so a turn that finished while the screen
  // was off leaves the mic held with no way to release it until now. Retrying here is what
  // stops a locked phone from ending the session.
  if (micHeld && handsfree && !speechPlaying && !speechQ.length) {
    await releaseMic("page visible again");
    setHfState("listening");
    return;
  }
  if (handsfree && !micHeld) {
    // the mic track is often dead even though the socket survived -- but in half-duplex it is
    // dead ON PURPOSE while the DM talks, and reviving it here moves the audio to the phone
    const live = stream && stream.getAudioTracks().some((t) => t.readyState === "live");
    if (!live) {
      closeMic(); source = null;
      if (await openMic()) { capturing = true; ws.send(JSON.stringify({ type: "listen" })); }
      else { handsfree = false; hf.classList.remove("on", "busy"); setHfState("listening"); }
    } else {
      ws.send(JSON.stringify({ type: "listen" }));      // re-arm server-side
    }
    setHfState("listening");
  }
}
document.addEventListener("visibilitychange", () => {
  report("visibility", document.visibilityState + (speechPlaying ? " (mid-speech)" : ""));
  wakeUp();
});
window.addEventListener("pagehide", (e) => report("pagehide", `persisted=${e.persisted} speaking=${speechPlaying}`));
window.addEventListener("freeze", () => report("freeze", "page frozen by the browser"));
window.addEventListener("resume", () => report("resume", "page unfrozen"));
document.addEventListener("visibilitychange", wakeUp);
window.addEventListener("focus", wakeUp);
window.addEventListener("pageshow", wakeUp);

// ---- updates land only when you ask for them ----
// Nothing here reloads on its own. Changes to the app sit on disk while you play; the button
// lights up to say something is waiting, and only a press applies it.
let pending = { client: false, server: false };

function markUpdate() {
  const on = pending.client || pending.server;
  const b = el("updateBtn");
  b.classList.toggle("on", on);
  b.textContent = on ? "⟳ Update ready" : "⟳ Update";
}
async function checkBuild() {
  try {
    const v = await api("/version");
    // Compare against what THIS PAGE actually loaded, not against what the server has in
    // memory — otherwise a phone running cached JS is told it's up to date.
    const mine = window.__BUILD__ || "";
    pending = {
      // No stamp at all means this page was cached before builds were stamped — treat it as
      // stale so one press fixes it, instead of forever reporting "already up to date".
      client: !!v.client_stale || !mine || (!!v.client && v.client !== mine),
      server: !!v.server_stale,
    };
    markUpdate();
    return pending.client || pending.server;
  } catch (e) { return false; }
}

// Safari will happily serve a cached bundle through an ordinary reload, so go to a URL it
// has never seen. The asset tags are stamped too, so the new page pulls fresh JS and CSS.
async function waitForServer(maxMs = 45000) {
  const until = Date.now() + maxMs;
  while (Date.now() < until) {
    try {
      const r = await fetch("/version", { cache: "no-store" });
      if (r.ok) return true;
    } catch (e) { /* still down */ }
    await new Promise((r) => setTimeout(r, 700));
  }
  return false;                       // reload anyway; better than hanging on this screen
}

function hardReload() {
  const u = new URL(location.href);
  u.searchParams.set("b", Date.now().toString(36));
  location.replace(u.toString());
}

// ...then take the cache-buster back out of the address bar, so what you see (and bookmark,
// and share) is the plain URL. The page is already loaded; this only rewrites the address.
(() => {
  const nav = (performance.getEntriesByType("navigation") || [])[0];
  report("page-load", `type=${nav ? nav.type : "?"} build=${window.__BUILD__ || "-"}`);
})();

if (location.search.includes("b=")) {
  const u = new URL(location.href);
  u.searchParams.delete("b");
  history.replaceState(null, "", u.pathname + (u.search || "") + u.hash);
}
setInterval(checkBuild, 20000);                    // only ever updates the badge

el("updateBtn").addEventListener("click", async () => {
  const b = el("updateBtn");
  b.textContent = "⟳ checking…";
  if (!(await checkBuild())) { addMsg("sys", "already on the latest build"); return; }
  if (pending.server) {
    b.textContent = "⟳ restarting…";
    addMsg("sys", "applying update — back in a few seconds");
    try { await api("/update", { method: "POST" }); } catch (e) {}
    // Wait for the server to actually be back rather than guessing. Six seconds was always a
    // guess, and the restart now recaps the session first, so a fixed timer would reload
    // straight into a server that is still coming up.
    waitForServer().then(hardReload);
  } else {
    b.textContent = "⟳ updating…";
    hardReload();
  }
});

el("campaignBtn").addEventListener("click", () => {
  panel.hidden = !panel.hidden;
  if (!panel.hidden) loadCampaigns();
});
el("cpClose").addEventListener("click", () => { panel.hidden = true; });
el("cpCreate").addEventListener("click", async () => {
  const name = el("cpName").value.trim();
  if (!name) { addMsg("sys", "give the campaign a name first"); return; }
  try {
    await api("/campaigns", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, premise: el("cpPremise").value.trim() }),
    });
    el("cpName").value = ""; el("cpPremise").value = "";
    switchTo(name);
  } catch (err) { addMsg("sys", "⚠ " + err.message); }
});

talk.disabled = true; hf.disabled = true;
connect();
loadCampaigns();

// the two controls that make a table of phones work at all
{
  const sb = el("seatBtn"); if (sb) sb.addEventListener("click", pickSeat);
  const pb = el("spkBtn"); if (pb) pb.addEventListener("click", toggleSpeaker);
}
{ const b = el("endTurn"); if (b) b.addEventListener("click", endMyTurn); }
