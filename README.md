# Dungeon Master

A real-time **voice** Dungeon Master. The **host PC** does all the work (speech-to-text,
Claude reasoning, text-to-speech on the GPU); your **phone** is the mic, speaker, and
control panel — just open a web page. Built to be driven remotely on your home network.

## Architecture
```
Phone browser  ──mic (16k PCM over WebSocket)──►  PC server
  (hold-to-speak, transcript, latency)              │  faster-whisper (STT, GPU)
        ▲                                            │  Sonnet (story) · qwen2.5:14b (combat) · Python (rules)
        └────────── DM voice (MP3) ◄────────────────┘  Kokoro-82M (voice, GPU)
```
Each stage is its own module (`server/stt.py`, `llm.py`, `tts.py`) so it can be swapped
local↔cloud as we tune latency and quality.

## This is the latency-harness prototype
Goal #1 is to **measure the round-trip on your hardware** before building game logic.
The phone shows three numbers per turn: STT time, DM time-to-first-token, and the one
that matters — **end-of-speech → first DM voice**. Target ≈ 1–2s.

## Setup (on the PC)
```bash
pip install -r requirements.txt
cp .env.example .env          # defaults are free + local; no API key needed
ollama pull qwen2.5:14b-instruct    # the local quick-answer model
bash certs/make_cert.sh       # one-time: self-signed cert (phone mic needs HTTPS)
python run.py                 # not server/app.py — see below
```

### Updates land only when you ask
Run it with **`run.py`**, not `server/app.py` directly. Edits to the server then sit inert on
disk while you play: `/version` reports them as pending and the **⟳ Update** button in the app
lights up, but nothing changes mid-session. Pressing it applies the update — the server exits,
`run.py` starts it again on the new code, and the phone reconnects. Measured at ~3s.
Client-only changes (JS/CSS/HTML) need no restart at all; the button just reloads the page.
First run downloads the whisper model (a few hundred MB). It logs the URL to open.

## Connect the phone (same WiFi)
1. Find the PC's LAN IP (`ipconfig` → IPv4, e.g. `192.168.1.42`).
2. On the phone browser go to **`https://192.168.1.42:8080`**.
3. It's a self-signed cert, so tap through the "Not Private" warning (**Advanced → proceed**).
   This is required — iOS only allows microphone access over HTTPS.
4. **Hold "Hold to Speak"**, talk, release. The DM replies in your ear, with timings on screen.

### Use the LAN. The tunnel is for players who are not in the room.

`cloudflared tunnel --url https://localhost:8080` gives a real HTTPS URL with no cert warning,
which makes it the tempting option — and a whole seven-player session ran through one while
every player sat at the same table on the same wifi. That evening opened **181 websockets with
a median lifetime of 54 seconds and 105 abnormal closures**. Utterances kept landing in sockets
that were gone before the DM could answer, which at the table looks exactly like the software
hanging.

A free quick tunnel is a development convenience, not transport for seven persistent
audio streams. Generate the cert, accept the warning once per phone, and talk to the machine
directly. `python start.py --status` prints the LAN address to use.

Keep the tunnel for the genuine case: somebody playing from another house.

## A table of phones

Each player opens the same address on their own phone and taps **🪑** to claim a character.
That makes "who said that" a fact rather than a voice-similarity score — enrolled voice
profiles at a real seven-player table sat up to 0.703 similar to one another and correctly
refused to name anybody.

- **🪑 Pick character** — one character per phone; the newest claim wins and the seat is
  remembered across a reload.
- **🔊 / 🔇** — which devices play the DM out loud. Put **one** phone next to the speakerphone
  and mute the rest, or seven phones narrate at once. If everybody mutes, the phone that asked
  still speaks, because silence everywhere reads as broken software.
- Maps, initiative, damage bars and every player's transcript go to **all** phones.
- **✔ End my turn** appears on the phone of whoever the engine is waiting on.

A player's turn ends when they say so — "that's my turn", "I'm done", "end turn" — or when they
tap the button. Nothing infers it: three sessions were spoiled by a turn ending on a question
or a movement. `AUTO_HANDOVER=1` restores the rule where another player acting ends the open
turn; it is off because the table asked for it off.

## Working on this

```bash
python -m pytest -q          # 248 tests, ~8s, no GPU or network needed
python start.py --status     # what is up, and the LAN address to open
python tools/playtest.py --help         # take turns without a microphone
python tools/concurrency_probe.py       # several players talking at once
python tools/narration_bench.py --help  # score prose against the reference corpus
```

`state/` is deliberately not in the repository: it holds live campaigns, session transcripts,
recorded audio and **voice embeddings tagged with players' real names**. That is one group's
data and biometrics, not part of the project. A fresh clone creates its own.

The tests are the main documentation of why things are the way they are. Most of them exist
because something failed at a real table, and each one says which failure it is guarding
against — that history is worth more than the assertions.

## Config (`.env`)
| var | default | notes |
|---|---|---|
| `DM_BACKEND` | `hybrid` | `hybrid` = local model answers lookups, Claude drives story. Also `claude_code`, `ollama`, `api` |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct` | the local quick-answer model (`ollama pull` it first) |
| `OLLAMA_KEEP_ALIVE` | `2h` | keeps it in VRAM; reloading a cold 14B costs ~9s |
| `ANTHROPIC_API_KEY` | — | only for `DM_BACKEND=api` |
| `DM_MODEL` | `claude-haiku-4-5-20251001` | fast; `claude-sonnet-5` = richer world-building, slower |
| `WHISPER_MODEL` | `large-v3-turbo` | on a 16GB GPU this is **both more accurate and ~2x faster** than `small.en` — no trade-off |
| `WHISPER_DEVICE` | `cuda` | `cpu` if no GPU |
| `WHISPER_BEAM` | `5` | decode beams; `1` is fastest but noticeably less accurate |
| `TTS_ENGINE` | `kokoro` | local neural voice on this machine; `edge` = Microsoft cloud fallback |
| `TTS_VOICE` | `am_michael` | kokoro voice id (`bm_george`, `af_heart`, …). Edge names only apply when `TTS_ENGINE=edge` |
| `TTS_SPEED` | `1.0` | kokoro rate; 1.0 is natural |
| `VAD_SILENCE_MS` | `1100` | pause before the DM answers — raise it if you get cut off |
| `VAD_NOISE_MULT` | `2.2` | lower = more sensitive to quiet speech |
| `VAD_CEILING` | `0.05` | threshold cap, so a loud moment can't deafen the mic |

## The DM brain (works with no audio)
The game logic is decoupled from the voice pipeline so it can be built and tested on its own:
- **`dice.py`** — real dice (`2d6+3`, `4d6kh3`, advantage). The DM never invents rolls.
- **`game_state.py`** — authoritative HP / AC / conditions / initiative / combat, persisted per campaign.
- **`tools.py`** — exposes dice + state as **Claude tools**; the DM *must* use them for anything
  mechanical, so numbers are real, not hallucinated.
- **`director.py`** — background **sub-agents** (Lorekeeper, Plot Weaver, NPC Director, Encounter
  Designer). The live DM stays fast; these run asynchronously between turns and write ideas/canon
  into the shared world state, so the *story* deepens without adding latency.

### Local rules cache (dense, offline, instant)
```bash
python server/srd_cache.py     # one-time: download the open-content library
```
Pulls ~5,100 open/CC/OGL entries (3,200 monsters, 1,400 spells, 1,600 magic items, plus
conditions, feats, rules sections…) into `state/cache/srd` and a name-keyed in-memory index.
Lookups are **~1 microsecond, offline** — the DM references rules with zero latency.
*Legal:* open-licensed content only; copyrighted books are never included.

### Remote play (any network) + hands-free
- **Off-network access:** `bash remote.sh` opens a public HTTPS tunnel (Cloudflare) so you can
  play from anywhere, not just the same WiFi — real cert, so the phone mic still works. Or use
  **Tailscale** for a private connection (phone reaches the PC by its tailnet IP from anywhere).
- **Auto-prompting (hands-free):** `vad.py` watches the audio energy; after you speak and pause,
  it fires end-of-utterance and prompts the DM automatically — no push-to-talk. The client sends
  `{"type":"listen"}` to enable it. The threshold **adapts to the room** (a fixed one either
  misses quiet speech or never hears the pause that ends your turn), and it returns the speech
  span itself — pre-roll included — so the first syllable survives.
- **Always listening — no re-arm delay.** The mic is *never* gated, so you can answer the
  instant the DM stops, or talk straight over it. The turn runs off the receive loop so the
  socket keeps draining while the DM thinks.
- **Barge-in by voice or by tap.** Talk over the DM and it stops and takes your correction; or
  hit ✋ Stop. Two things keep it from interrupting *itself*, since its own voice is in the
  open mic: the VAD demands a louder, longer burst while the DM holds the floor, and a
  barge-in is only honoured after the audio is transcribed and checked against what the DM
  just said (`_is_echo`). It stops *speaking* at once but still drains the response — the
  Claude Code session is one long-lived process, so abandoning it mid-answer would leak the
  tail into the next turn.
- **Pipelined speech.** Each sentence goes to the voice engine as soon as the DM writes it,
  while earlier ones are still playing — synthesis is a 0.5-2s round trip, so doing it in
  lockstep left dead air between every sentence.

### Measured response time (end of your speech → first DM voice)
Real numbers over the websocket with synthesized speech, `claude_code` backend, warm session:

| stage | cost | notes |
|---|---|---|
| end-of-turn pause | 1100ms | `VAD_SILENCE_MS`; shorter cuts you off mid-sentence |
| speech recognition | **0ms** | runs *during* the pause (see below), off the critical path |
| DM thinking — **field read** | **~0.1µs** | pure Python, no model (`quick.py`) |
| DM thinking — **lookup** | ~0.5s | local `qwen2.5:14b` on the GPU |
| DM thinking — **story** | 3.5–7s | the Claude Code CLI; the remaining latency floor |
| voice | ~400ms | local kokoro, consistent (edge-tts was 500–2100ms, network-dependent) |

**"What's my Perception modifier?" lands in ~1.5s — nearly all of it the deliberate
end-of-turn pause; the server's own work is 41ms once the audio is cached.** A story beat is
6–10s, almost all of it Claude.

### Three tiers, cheapest first
Most of what a player says at the table isn't a story beat — it's a lookup. Each tier only
handles what it can do *correctly*, and hands on anything else.

| tier | handles | cost |
|---|---|---|
| **`quick.py` — pure Python** | AC, HP, ability & skill modifiers, initiative, conditions, whose turn, "I take 6 damage", "next turn" | **~0.1µs** |
| **local `qwen2.5:14b`** | mechanical questions the code can't parse | ~500ms |
| **Claude** | everything that moves the world | 3.5–7s |

The Python tier matters as much for **correctness** as speed: a model asked "what's my AC"
can state a number that isn't on your sheet and say it with total confidence. `combatant.ac`
cannot. Every pattern is a **fullmatch** against the whole utterance, so `what's my AC` is
answered here and `what's my AC against the ogre's bite` isn't matched at all and falls
through — with no model in the loop there's nothing to catch a misparse, so anything less
than certain must be handed on. Same for *whose* sheet: with two PCs and no speaker
identified, "my AC" is ambiguous and goes to a model rather than guessing.

Skill modifiers are only answered when the sheet's **proficiencies** are known — a proficient
level 1 druid's Perception is +5, not the +3 the raw Wisdom score implies, so without that
data the code declines rather than stating a confident wrong number. `ddb.py` imports
proficiency and expertise from D&D Beyond for exactly this reason.

Deterministic answers repeat all session ("Plus 5.", "15."), so synthesized audio is cached
by (engine, voice, text): a repeated question is answered and spoken in **~41ms** server-side.

### Two brains (`DM_BACKEND=hybrid`)
Mechanical lookups — modifiers, AC, HP, whose turn, spell slots — are already answered by the
game state, so they go to a local 14B that replies in ~0.5s and never touches the story.
Everything that moves the world goes to Claude. The routing is deliberately lopsided: a
question only takes the fast path if it matches a narrow question shape AND a mechanical
topic AND isn't an action; anything ambiguous goes to Claude, and if Ollama is down the
quick path silently falls back to it.

Everything that *can* be taken off the critical path has been:
- **Speculative recognition** — transcription starts as soon as you pause, running during the
  1.1s that decides whether you're actually done. If you resume, the guess is discarded.
- **Clause-level streaming** — narration is spoken at the first clause, not the first full
  sentence, so a long opening line is no longer dead air.
- **Pipelined + local speech** — no network round trip, and sentence N+1 is synthesized while
  N is still playing.

**The remaining wall is the DM's first token**, and it is the price of the free Claude Code
backend. The two ways down: `DM_BACKEND=api` with an Anthropic key (fast, bills per token), or
a local open-weights model on the GPU (free and fast, at some cost in DM quality).

```bash
python sim_latency.py          # models the stages + network for other configurations
```

### External connectors
- **`rules.py`** — rules lookup over the free, **open SRD** APIs (dnd5eapi.co + open5e):
  spells, monster stat blocks, conditions. Cached to disk. Exposed as the `lookup_rule`
  tool, plus `spawn_monster` (adds real-stat SRD monsters to a fight).
  *Legal boundary:* only SRD / open-licensed content is available via API — the full
  PHB/MM/etc. are copyrighted and can't be pulled from anywhere. This is core rules + SRD.
- **`ddb.py`** — imports a **D&D Beyond** character sheet by id/URL (`import_character` tool).
  DDB has no official API, so it uses the character-service JSON endpoint the ecosystem
  relies on — which requires the sheet's **sharing to be set to "Public."** HP/stats import
  cleanly; final AC is DDB-computed from gear (not exposed), so it's approximated and can be
  corrected in-game.

### Play / test by text (no phone, no latency pipeline)
```bash
python play.py my-campaign      # type actions; DM rolls dice + tracks combat live
python tests/test_core.py       # deterministic tests (dice, combat, tools, director) — no API key
```
`play.py` runs the full brain over text. With `ANTHROPIC_API_KEY` set you get real DM narration
and sub-agent world-building; without it, mechanics + mock director notes still work.

## Roadmap
- [x] Phone ↔ PC voice round-trip, push-to-talk, per-stage latency
- [x] Persistent world/campaign state, NPCs, initiative, combat, dice (tool-grounded)
- [x] Background sub-agent Director for story/world-building
- [x] Text-mode play + test harness (develop the brain without audio)
- [x] SRD rules/spell/monster/condition lookup (open APIs) + D&D Beyond character import
- [x] Dense offline open-content cache (~5,100 entries, ~1µs lookups)
- [x] Remote play over the internet (tunnel) + hands-free auto-prompting (VAD)
- [x] Response-time simulation harness
- [x] Custom vocabulary biasing for fantasy names (campaign proper nouns prime the decoder)
- [x] Adaptive hands-free VAD, half-duplex turn-taking, tap-to-interrupt barge-in
- [ ] Streaming STT (transcribe while speaking) to cut latency
- [ ] Voice-triggered barge-in (talk over the DM) — needs echo cancellation, not just a button
- [ ] Local low-latency TTS (Piper) / premium voice (ElevenLabs) options

## Independent project
No relationship to anything else on this machine. See `CLAUDE.md`.
