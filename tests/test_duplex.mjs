// The half-duplex microphone handover, driven through the sequences that actually broke it.
//
// This exists because the handover was shipped three times on reasoning alone and was wrong
// three times -- each fix looked obviously right and each one failed in the room, on a phone,
// twenty minutes into a session, where the only evidence was a line in a log. The state machine
// is small and its inputs are known, so there is no excuse for not driving it.
//
// It runs the REAL web/app.js inside a vm context with a fake DOM, a fake microphone and a
// virtual clock, rather than a copy of the logic: a test that agrees with a reimplementation
// proves nothing about the file the phone loads. Top-level `let` bindings in a vm script stay
// reachable from later evaluations in the same context, which is what makes the internal state
// observable without exporting anything for the test's benefit.
import { test } from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const APP = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web", "app.js");

function makeEnv({ search = "?duplex=half", micWorks = () => true,
                   mic = "iPhone Microphone" } = {}) {
  let now = 1_000_000;
  const timers = [];              // {id, due, fn, every}
  let nextId = 1;
  const log = [];                 // every report() the client makes
  const opens = [];               // every getUserMedia call

  const listeners = {};
  const elFor = (id) => new Proxy({ id, style: {}, dataset: {}, classList: {
    add() {}, remove() {}, toggle() {}, contains: () => false } }, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === "addEventListener" || k === "removeEventListener") return () => {};
      if (k === "appendChild" || k === "remove" || k === "focus" || k === "scrollIntoView"
          || k === "querySelectorAll" || k === "querySelector" || k === "insertBefore") {
        return k.startsWith("query") ? () => (k === "querySelectorAll" ? [] : null) : () => {};
      }
      if (k === "textContent" || k === "innerHTML" || k === "value") return "";
      if (k === "hidden" || k === "disabled") return false;
      return undefined;
    },
    set(t, k, v) { t[k] = v; return true; },
  });

  function track(live = true) {
    const t = { kind: "audio", readyState: live ? "live" : "ended", label: mic,
                enabled: true, muted: false, stop() { t.readyState = "ended"; },
                getSettings: () => ({ deviceId: "default" }) };
    return t;
  }

  const ctxState = { state: "running" };
  const sandbox = {
    console,
    Date: new Proxy(Date, { get: (t, k) => (k === "now" ? () => now : t[k]) }),
    setTimeout: (fn, ms) => { const id = nextId++; timers.push({ id, due: now + (ms || 0), fn }); return id; },
    clearTimeout: (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); },
    setInterval: (fn, ms) => { const id = nextId++; timers.push({ id, due: now + ms, fn, every: ms }); return id; },
    clearInterval: (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); },
    location: { protocol: "https:", host: "x.test", search, href: "https://x.test/" + search },
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    URL, URLSearchParams, Blob: class { constructor() {} }, TextEncoder, TextDecoder,
    fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }),
    performance: { now: () => now, getEntriesByType: () => [{ type: "navigate" }] },
    btoa: (x) => Buffer.from(x, "binary").toString("base64"),
    atob: (x) => Buffer.from(x, "base64").toString("binary"),
    Buffer,
    navigator: {
      sendBeacon: (_u, b) => { log.push(String(b && b._body || "")); return true; },
      mediaDevices: {
        enumerateDevices: () => Promise.resolve([{ kind: "audioinput", deviceId: "default",
                                                   label: mic }]),
        getUserMedia: () => {
          opens.push(now);
          if (!micWorks()) return Promise.reject(Object.assign(new Error("no"), { name: "NotAllowedError" }));
          const ts = [track()];
          return Promise.resolve({ getTracks: () => ts, getAudioTracks: () => ts });
        },
      },
      wakeLock: { request: () => Promise.reject(new Error("nope")) },
    },
    WebSocket: class {
      constructor() { this.readyState = 1; setTimeout(() => this.onopen && this.onopen({}), 0); }
      send() {} close() { this.readyState = 3; }
    },
    Audio: class { constructor() { this.paused = true; }
      setAttribute() {} addEventListener(k, f) { this["_" + k] = f; }
      play() { this.paused = false; return Promise.resolve(); }
      pause() { this.paused = true; } load() {} },
    AudioContext: class {
      constructor() { Object.assign(this, ctxState); this.sampleRate = 48000; this.state = "running"; }
      resume() { this.state = "running"; return Promise.resolve(); }
      suspend() { this.state = "suspended"; return Promise.resolve(); }
      close() { this.state = "closed"; return Promise.resolve(); }
      createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
      createMediaStreamDestination() { return { stream: {} }; }
      createGain() { return { gain: {}, connect() {}, disconnect() {} }; }
      createScriptProcessor() { return { connect() {}, disconnect() {}, onaudioprocess: null }; }
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.webkitAudioContext = sandbox.AudioContext;
  sandbox.document = {
    visibilityState: "visible",
    getElementById: elFor, querySelector: () => elFor("q"), querySelectorAll: () => [],
    createElement: elFor, body: elFor("body"), documentElement: elFor("html"),
    addEventListener: (k, f) => { (listeners[k] = listeners[k] || []).push(f); },
    removeEventListener() {},
  };
  sandbox.addEventListener = (k, f) => { (listeners[k] = listeners[k] || []).push(f); };
  sandbox.removeEventListener = () => {};
  // URL.createObjectURL is how the client hands a clip to the <audio> element
  sandbox.URL = Object.assign(function (u) { return new URL(u); },
    { createObjectURL: () => "blob:x", revokeObjectURL() {} });

  const ctx = vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(APP, "utf8"), ctx, { filename: "app.js" });

  const drain = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
  return {
    ctx, opens, log, listeners,
    get: (expr) => vm.runInContext(expr, ctx),
    run: (src) => vm.runInContext(src, ctx),
    async advance(ms) {
      const target = now + ms;
      for (let guard = 0; guard < 10000; guard++) {
        const due = timers.filter((t) => t.due <= target).sort((a, b) => a.due - b.due)[0];
        if (!due) break;
        now = due.due;
        if (due.every) due.due = now + due.every;
        else timers.splice(timers.indexOf(due), 1);
        try { due.fn(); } catch (e) { /* a timer that throws is the client's problem, not ours */ }
        await drain();
      }
      now = target;
      await drain();
    },
    async settle() { await drain(); },
    fire(name) { (listeners[name] || []).forEach((f) => f({})); },
    reports: () => log.map((s) => { try { return JSON.parse(s); } catch (e) { return { detail: s }; } }),
  };
}

// The client posts diagnostics through sendBeacon(Blob), and the Blob body is not readable in
// this shim -- so capture report() at the source instead. Same data, no guessing.
function withReports(env) {
  env.run(`globalThis.__reports = [];
           report = (e, d) => { __reports.push(e + ": " + d); };`);
  return () => env.get("__reports.slice()");
}

/** Put the client into the state a real turn starts from: connected, hands-free, mic open. */
async function armed(env) {
  env.run("handsfree = true; capturing = true; turnOver = true;");
  env.run("stream = { getAudioTracks: () => [{ readyState: 'live', label: 'iPhone Microphone' }],"
        + " getTracks: () => [{ stop(){} }] };");
  env.run("source = { connect(){}, disconnect(){} }; processor = { connect(){}, disconnect(){}, onaudioprocess: null };");
  await env.settle();
}

/** One sentence of narration arriving from the server. */
async function sentence(env) {
  env.run("speechQ.push('blob:x'); if (!speechPlaying) _nextSpeech();");
  await env.settle();
}
/** That sentence finishing playing. */
async function sentenceEnded(env) {
  env.run("_nextSpeech();");
  await env.settle();
}

test("half-duplex is on when the URL asks for it", async () => {
  const env = makeEnv();
  assert.equal(env.get("HALF_DUPLEX"), true);
  assert.equal(makeEnv({ search: "" }).get("HALF_DUPLEX"), false);
});

test("the mic is shut once per turn, not once per sentence", async () => {
  const env = makeEnv();
  const reports = withReports(env);
  await armed(env);
  env.run("turnOver = false;");

  await sentence(env);
  assert.equal(env.get("micHeld"), true, "first clip must close the mic");
  const held = env.get("_heldSince");

  // three more sentences, each with the queue running dry in between -- the gap that used to
  // unlatch every guard keyed on speechPlaying
  for (let i = 0; i < 3; i++) {
    await sentenceEnded(env);
    await env.advance(300);
    await sentence(env);
    assert.equal(env.get("micHeld"), true, `sentence ${i + 2}: mic must STAY held`);
  }
  assert.equal(env.get("_heldSince"), held, "the hold clock must not restart per sentence");
  assert.equal(reports().filter((r) => r.includes("mic closed")).length, 1);
});

test("a long turn does not hand the mic back in the middle of itself", async () => {
  // The fallback deadline measures silence, not the length of the narration. Measured from the
  // wrong clock, any turn longer than the timeout reopens mid-sentence and the audio jumps back
  // to the phone -- which is the bug this test is here to keep out.
  const env = makeEnv();
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  for (let i = 0; i < 12; i++) {          // ~24 seconds of narration, well past the 4s deadline
    await env.advance(2000);
    assert.equal(env.get("micHeld"), true, `still speaking at ${(i + 1) * 2}s`);
  }
  assert.equal(env.get("micHeld"), true);
});

test("the mic comes back when the turn ends", async () => {
  const env = makeEnv();
  const reports = withReports(env);
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  await sentenceEnded(env);
  env.run("turnOver = true;");
  await env.advance(2000);
  assert.equal(env.get("micHeld"), false, "turn over: the mic must be back");
  assert.equal(env.get("capturing"), true);
  assert.ok(reports().some((r) => r.includes("mic reopened")), reports().join("\n"));
});

test("a missing turn_end does not cost the session its microphone", async () => {
  // "It stops after the first message it sends": the reopen waited on a server message that
  // never came, so the mic stayed shut for the rest of the session and the DM went deaf.
  const env = makeEnv();
  const reports = withReports(env);
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  await sentenceEnded(env);
  await env.advance(1000);
  assert.equal(env.get("micHeld"), true, "one second of quiet is just a gap between sentences");
  await env.advance(6000);
  assert.equal(env.get("micHeld"), false, "silence has to be enough on its own");
  assert.ok(reports().some((r) => /mic reopened.*no audio for/.test(r)), reports().join("\n"));
});

test("a screen that locks mid-narration does not end the session", async () => {
  // iOS will not give a hidden page a capture device, so a turn that finishes while the phone
  // is face-down leaves the mic held with nothing able to release it. Coming back to the front
  // is the first moment it can be, and it has to be taken.
  let visible = true;
  const env = makeEnv({ micWorks: () => visible });
  const reports = withReports(env);
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);

  visible = false;
  env.run("document.visibilityState = 'hidden';");
  await sentenceEnded(env);
  env.run("turnOver = true;");
  await env.advance(15000);
  assert.equal(env.get("micHeld"), true, "hidden: nothing to do but wait");

  visible = true;
  env.run("document.visibilityState = 'visible';");
  env.fire("visibilitychange");
  await env.advance(1500);
  assert.equal(env.get("micHeld"), false, "back in front: the mic must return");
  assert.ok(reports().some((r) => /mic reopened/.test(r)), reports().join("\n"));
});

test("the watchdog leaves a deliberately-closed mic alone while the DM talks", async () => {
  // The watchdog rebuilds a mic that iOS killed. It cannot tell that from one closed on
  // purpose, and every rebuild puts iOS back into PlayAndRecord -- the audio hopping off the
  // Bluetooth speaker mid-sentence, on the watchdog's own 2.5-second cycle.
  const env = makeEnv();
  const reports = withReports(env);
  await armed(env);
  env.run("micTimer = setInterval(() => {}, 2500);");   // placeholder; real one installed below
  env.run("clearInterval(micTimer); toggleHF && 0;");
  env.run("turnOver = false;");
  await sentence(env);
  await env.advance(12000);
  assert.equal(env.get("micHeld"), true);
  assert.ok(!reports().some((r) => r.includes("mic-dead")),
            "nothing may rebuild the mic while it is held: " + reports().join("\n"));
});

test("hold-to-speak takes the mic back mid-narration", async () => {
  const env = makeEnv();
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  assert.equal(env.get("micHeld"), true);
  await env.run("startRec()");
  await env.settle();
  assert.equal(env.get("micHeld"), false, "the player asked for the mic; it is theirs");
  assert.equal(env.get("recording"), true);
});

test("stopping the DM brings the mic back", async () => {
  const env = makeEnv();
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  env.run("stopSpeech();");
  await env.advance(1500);
  assert.equal(env.get("micHeld"), false);
});

test("turning hands-free off releases the hold", async () => {
  const env = makeEnv();
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  env.run("handsfree = false; cancelMicReopen(); micHeld = false;");
  await env.advance(8000);
  assert.equal(env.get("micHeld"), false);
  assert.equal(env.get("_reopenTimer"), 0, "no timer may outlive hands-free");
});

test("a speakerphone turns half-duplex off by itself", async () => {
  // The Anker PowerConf and anything like it present a microphone AND a speaker, so iOS can
  // run both directions at once and the whole closed-mic workaround is obsolete -- keeping it
  // would cost barge-in for nothing. The URL asks for half; the hardware overrules it.
  const env = makeEnv({ search: "?duplex=half", mic: "Anker PowerConf (Bluetooth)" });
  const reports = withReports(env);
  assert.equal(env.get("HALF_DUPLEX"), true, "the URL is still the opening position");
  await env.run("openMic()");
  await env.settle();
  assert.equal(env.get("HALF_DUPLEX"), false, "a speakerphone must switch it off");
  assert.ok(reports().some((r) => /full duplex/.test(r)), reports().join(String.fromCharCode(10)));

  // ...and the mic then stays open through a whole turn, which is the point of doing it
  await armed(env);
  env.run("turnOver = false;");
  await sentence(env);
  assert.equal(env.get("micHeld"), false, "nothing may close the mic in full duplex");
  assert.equal(env.get("capturing"), true, "the table can talk over the DM again");
});

test("a speaker with no microphone keeps half-duplex", async () => {
  const env = makeEnv({ search: "?duplex=half", mic: "iPhone Microphone" });
  await env.run("openMic()");
  await env.settle();
  assert.equal(env.get("HALF_DUPLEX"), true, "output-only speaker: the workaround still earns its place");
});
