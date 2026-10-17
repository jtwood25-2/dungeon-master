"""Dungeon Master server: phone <-> PC pipeline (STT -> Claude -> TTS), instrumented.

Protocol over /ws:
  client -> server : {"type":"start"} | <binary Int16LE PCM @16k> | {"type":"end"}
                     {"type":"listen"|"stop_listen"}          hands-free on/off
                     {"type":"playing","on":bool}             is the phone speaker active
                     {"type":"interrupt"}                     cut the DM off now
  server -> client : {"type":"status"|"transcript"|"dm"|"timing", ...} | <binary MP3>
                     {"type":"turn_start"|"turn_end"|"interrupted"}

In hands-free the mic is NEVER gated: the player can speak the instant the DM stops, or over
the top of it. Barge-in is confirmed by transcript before the DM is cut off, so its own echo
can't interrupt it.
"""
import asyncio, json, logging, os, re, time, uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config, stt, llm, tts, quick, feedback, safety, sheet, game_state, director as director_mod
import campaign, planner, transcript, endpoint, backend as backend_mod
import ddb
import voices as voices_mod, speakers, ambience, maps
import audiolog
import turns
import room as room_mod
import recap as recap_mod
from vad import VAD
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("dm")

app = FastAPI(title="Dungeon Master")
WEB = Path(__file__).parent.parent / "web"


@app.on_event("startup")
async def _startup():
    await asyncio.to_thread(stt.warmup)          # load whisper before the first player speaks
    await tts.warmup()                            # and the voice model, so line one isn't slow
    try:
        await llm.prewarm()                       # boot the DM (warm Claude Code session) up front
    except Exception as e:
        logger.warning(f"DM prewarm skipped: {e}")
    logger.info(f"ready on http://{config.HOST}:{config.PORT}  (open this on your phone)")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the page with the build stamped into the asset URLs.

    Without this, updates silently didn't land: the assets were referenced as plain
    /static/app.js with no Cache-Control, so Safari kept serving a cached copy and even
    location.reload() wouldn't refetch it. A new build is now a new URL, and the page itself
    is never cached. The stamp is also handed to the client so it can tell what it's running
    — /version only ever knew what the SERVER had, so it would report "up to date" while the
    phone was on old code.
    """
    stamp = _stamp(CLIENT_EXT)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
    html = html.replace("/static/style.css", f"/static/style.css?v={stamp}")
    html = html.replace("</head>", f'<script>window.__BUILD__="{stamp}";</script></head>')
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """Paste-a-link table setup: import sheets and seat players without using the mic."""
    return HTMLResponse((WEB / "setup.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/party/{name}")
async def party(name: str):
    """Who's in this campaign, for the sheet picker."""
    g = game_state.GameState.load(campaign._safe(name))
    return {"party": [{"name": c.name, "hp": c.hp, "max_hp": c.max_hp, "ac": c.ac,
                       "level": getattr(c, "level", 1), "notes": c.notes}
                      for c in g.combatants.values() if c.is_pc]}


@app.get("/sheet/{name}", response_class=HTMLResponse)
async def sheets(name: str, who: str = ""):
    """Printable character sheet(s) — the whole party, or one character via ?who=."""
    g = game_state.GameState.load(campaign._safe(name))
    pcs = [c for c in g.combatants.values() if c.is_pc]
    if who:
        pcs = [c for c in pcs if c.name.lower() == who.lower()]
        if not pcs:
            raise HTTPException(404, "no such character")
    return HTMLResponse(sheet.page(pcs, g.campaign))


@app.get("/maps/{name}")
async def maps_list(name: str):
    """Every map this campaign has ever drawn — they persist until cleared."""
    return {"maps": maps.listing(campaign._safe(name))}


@app.get("/maps/{name}/{slug}", response_class=HTMLResponse)
async def maps_view(name: str, slug: str):
    """A stored map on its own page, fully revealed — for looking back or printing."""
    svg = maps.svg_of(campaign._safe(name), slug, lambda m: _render_map(m, reveal_all=True))
    if not svg:
        raise HTTPException(404, "no such map")
    return HTMLResponse(f"<!doctype html><meta name=viewport content='width=device-width,"
                        f"initial-scale=1'><title>{slug}</title>"
                        f"<body style='margin:0;background:#12101a'>{svg}</body>")


@app.delete("/maps/{name}")
async def maps_clear(name: str, slug: str = ""):
    """Clear one map (?slug=) or the campaign's whole library."""
    n = maps.clear(campaign._safe(name), slug or None)
    return {"cleared": n}


@app.get("/feedback")
async def feedback_list():
    return {"items": feedback.items()}


@app.post("/feedback")
async def feedback_add(body: dict):
    note = (body or {}).get("text", "").strip()
    if not note:
        raise HTTPException(400, "text required")
    return feedback.add(note, (body or {}).get("campaign", ""), 0)


@app.post("/feedback/{item_id}/{status}")
async def feedback_status(item_id: str, status: str):
    if status not in (feedback.OPEN, feedback.DONE, "deleted"):
        raise HTTPException(400, "bad status")
    if not feedback.set_status(item_id, status):
        raise HTTPException(404, "no such note")
    return {"id": item_id, "status": status}


def _stamp(exts) -> str:
    newest = 0.0
    for d in (WEB, Path(__file__).parent):
        for p in d.glob("*.*"):
            if p.suffix in exts:
                newest = max(newest, p.stat().st_mtime)
    # .env is server config and changing it changes behaviour, but it was invisible here:
    # switching the narrator voice left the button reporting "up to date" while the running
    # process still held the old value, and only a manual restart applied it.
    if ".py" in exts:
        env = Path(__file__).parent.parent / ".env"
        if env.exists():
            newest = max(newest, env.stat().st_mtime)
    return f"{newest:.0f}"


CLIENT_EXT = (".js", ".css", ".html")
SERVER_EXT = (".py",)
RUNNING = {"client": _stamp(CLIENT_EXT), "server": _stamp(SERVER_EXT)}
LIVE: dict = {}                 # sid -> GameState, so an update can save what's in play


@app.post("/clientlog")
async def clientlog(body: dict):
    """The phone reporting what happened to it. The server log can see a socket close but not
    WHY -- a reload, a discarded tab, a frozen page and a dropped link look identical from
    here, and each has a different fix."""
    ev = str((body or {}).get("event", ""))[:40]
    detail = str((body or {}).get("detail", ""))[:300]
    logger.info(f"[phone] {ev}: {detail}")
    return {"ok": True}


@app.get("/link")
async def link():
    """The current public URL, for sharing with someone else at the table."""
    p = Path(__file__).parent.parent / "state" / "tunnel-url.txt"
    return {"url": p.read_text(encoding="utf-8").strip() if p.exists() else ""}


@app.get("/version")
async def version():
    """What's running vs what's on disk. Edits sit inert until the player asks for them —
    nothing about their session changes on someone else's schedule."""
    disk_client, disk_server = _stamp(CLIENT_EXT), _stamp(SERVER_EXT)
    return {"version": disk_client + "." + disk_server,
            "client": disk_client,           # what the phone SHOULD be running
            "client_stale": disk_client != RUNNING["client"],
            "server_stale": disk_server != RUNNING["server"]}


@app.post("/update")
async def update():
    """Apply pending server changes: exit so run.py restarts us on the new code. Only ever
    called by the player pressing the button."""
    if _stamp(SERVER_EXT) == RUNNING["server"]:
        return {"restarting": False, "reason": "server already up to date"}

    async def bye():
        await asyncio.sleep(0.4)                 # let the response reach the phone first
        # os._exit skips every finally block, so anything in play must be written down HERE
        # or applying an update would quietly lose the scene it was applied in.
        for st in list(LIVE.values()):
            try:
                # The recap normally happens in the websocket's finally block, which os._exit
                # skips — so an update applied mid-session used to preserve the STATE and
                # leave the "previously on" stale, describing a scene two beats ago.
                try:
                    await asyncio.wait_for(planner.recap_session(st), timeout=12)
                except Exception as e:
                    logger.warning(f"recap before restart failed ({e}); state still saved")
                st.save()                        # save() stamps last_played itself
                logger.info(f"saved + recapped {st.campaign} (session {st.session}) before restart")
            except Exception as e:
                logger.warning(f"save before restart failed: {e}")
        logger.info("update requested — restarting on new code")
        os._exit(3)                              # run.py sees 3 and starts us again
    asyncio.create_task(bye())
    return {"restarting": True}


@app.get("/campaigns")
async def campaigns():
    return {"campaigns": campaign.list_campaigns(), "current": campaign.best_default()}


@app.get("/campaigns/{name}")
async def campaign_detail(name: str):
    d = campaign.detail(name)
    if d is None:
        raise HTTPException(404, "no such campaign")
    return d


@app.post("/campaigns")
async def campaign_create(body: dict):
    name = (body or {}).get("name", "").strip()
    if not name:
        raise HTTPException(400, "name required")
    if campaign.detail(name) is not None:
        raise HTTPException(409, "a campaign with that name already exists")
    return campaign.create(name, (body or {}).get("premise", ""))


@app.delete("/campaigns/{name}")
async def campaign_delete(name: str):
    if not campaign.delete(name):
        raise HTTPException(404, "no such campaign")
    return {"deleted": name}


@app.post("/campaigns/{name}/new-session")
async def campaign_new_session(name: str):
    """Close tonight's session without losing anything; the next connect opens a new one."""
    if not campaign.new_session(name):
        raise HTTPException(404, "no such campaign")
    for sid, st in list(LIVE.items()):        # a live copy would write the old stamp straight back
        if st.campaign == campaign._safe(name):
            st.last_played = 0.0
    # Write and voice the "previously on" NOW, while the room is empty. It is the same work
    # either way; the only question is whether the table pays for it. Opus writes it because
    # latency is free here, and it is rendered to audio in the same pass.
    import game_state as gs_mod
    st = next((x for x in LIVE.values() if x.campaign == campaign._safe(name)), None)         or gs_mod.GameState.load(campaign._safe(name))
    asyncio.create_task(_build_recap(st))
    return {"new_session": name, "recap": "building"}


async def _build_recap(state) -> None:
    """Background, and never allowed to take anything else down with it."""
    try:
        await recap_mod.build(state)
    except Exception as e:
        logger.warning(f"recap build failed ({e}); the next session will recap live")


@app.post("/campaigns/{name}/reset")
async def campaign_reset(name: str):
    if not campaign.reset(name):
        raise HTTPException(404, "no such campaign")
    return {"reset": name}


def _editable(name: str):
    """The GameState to write roster changes into.

    If the campaign is being played right now, the live copy is authoritative — editing a
    disk copy would be silently overwritten the next time that session saves. Returns
    (state, is_live); the caller only saves when it isn't live.
    """
    safe = campaign._safe(name)
    for st in LIVE.values():
        if st.campaign == safe:
            return st, True
    return game_state.GameState.load(safe), False


@app.get("/campaigns/{name}/players")
async def players_list(name: str):
    if campaign.detail(name) is None:
        raise HTTPException(404, "no such campaign")
    state, _ = _editable(name)
    out = []
    for r in state.roster:
        c = state.get(r.get("character", ""))
        out.append({**r, "level": getattr(c, "level", 0) if c else 0,
                    "notes": getattr(c, "notes", "") if c else "",
                    "imported": bool(c and getattr(c, "ddb_id", ""))})
    return {"campaign": name, "players": out}


@app.post("/campaigns/{name}/players")
async def players_add(name: str, body: dict):
    """Import a D&D Beyond sheet and seat that player, from a pasted link.

    The only route in before this was the DM's [IMPORT 154114558] action, which means
    reading a nine-digit id out loud to a microphone -- fine for one returning character,
    unusable for a table of new players.
    """
    if campaign.detail(name) is None:
        raise HTTPException(404, "no such campaign")
    player = str((body or {}).get("player", "")).strip()
    sheet = str((body or {}).get("sheet", "")).strip()
    if not player or not sheet:
        raise HTTPException(400, "player and sheet are both required")

    state, live = _editable(name)
    msg = ddb.import_to_state(sheet, state)
    if msg.startswith("Import failed"):
        raise HTTPException(400, msg)

    char = ddb._char_id(sheet)
    who = next((c.name for c in state.combatants.values()
                if getattr(c, "ddb_id", "") == char), "")
    if not who:
        raise HTTPException(400, "imported, but the character could not be identified")
    state.add_player(player, who, ddb_id=char)
    if not live and not state.save():
        raise HTTPException(500, "the write was refused to protect the campaign file; "
                                 "see the REFUSED line in the server log")
    return {"player": player, "character": who, "detail": msg, "live": live}


@app.delete("/campaigns/{name}/players/{player}")
async def players_remove(name: str, player: str):
    """Drop a seat — for the wrong link pasted, or someone who bailed before session one."""
    if campaign.detail(name) is None:
        raise HTTPException(404, "no such campaign")
    state, live = _editable(name)
    row = next((r for r in state.roster if r.get("player") == player), None)
    if row is None:
        raise HTTPException(404, f"{player} is not on this roster")
    state.roster = [r for r in state.roster if r.get("player") != player]
    state.combatants.pop((row.get("character") or "").lower(), None)
    # Dropping the last PC is indistinguishable from the accident the regression guard exists
    # to stop, so it refuses the write and the endpoint would have reported a success that
    # never reached disk. This removal IS the deliberate one the guard asks for.
    if not live and not state.save(force=True):
        raise HTTPException(500, "the write was refused; see the server log")
    return {"removed": player, "character": row.get("character", "")}


@app.post("/campaigns/{name}/resync")
async def players_resync(name: str):
    """Re-read every imported sheet. Run this after a level-up night."""
    if campaign.detail(name) is None:
        raise HTTPException(404, "no such campaign")
    state, live = _editable(name)
    out = []
    for c in list(state.combatants.values()):
        if not (c.is_pc and getattr(c, "ddb_id", "")):
            continue
        r = ddb.resync(c)
        out.append({"character": c.name, **r})
    if not live and not state.save():
        raise HTTPException(500, "the write was refused; see the server log")
    return {"resynced": out}


class _Revalidating(StaticFiles):
    """Assets must be revalidated, never served blind from cache. Without this Safari kept a
    stale bundle and an update looked like it simply hadn't happened."""

    def file_response(self, *a, **kw):
        resp = super().file_response(*a, **kw)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/static", _Revalidating(directory=WEB), name="static")


class _Phone:
    """The phone can vanish mid-sentence — locked, backgrounded, tunnel blinked — and writing
    into a closed socket raises out of the whole handler.

    I guarded the send sites one at a time and missed one: _cold_open sends the recap TEXT and
    then its audio, so the recap appeared on screen and nothing ever spoke. There are 31 send
    calls; guarding them individually is a bug waiting to recur, so the socket itself is what
    tolerates a client that has gone.
    """
    def __init__(self, sock: WebSocket, sid: str = ""):
        self._sock, self._sid, self.gone = sock, sid, False

    async def send_text(self, data):
        await self._send(self._sock.send_text, data)

    async def send_bytes(self, data):
        await self._send(self._sock.send_bytes, data)

    async def _send(self, fn, data):
        if self.gone:
            return
        try:
            await fn(data)
        except (RuntimeError, WebSocketDisconnect):
            self.gone = True
            logger.info(f"[{self._sid}] phone left; dropping the rest of this turn")

    def __getattr__(self, name):        # receive() etc. pass through and still raise
        return getattr(self._sock, name)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    sock = _Phone(sock)
    sid = uuid.uuid4().hex[:8]
    sock._sid = sid
    buf = bytearray()
    try:
        state = campaign.start_session(campaign.best_default())  # resumes if we were just playing
    except FileNotFoundError:
        # genuinely nothing on disk yet (first run) -- creating one here is right. Reaching
        # for a campaign that was DELETED is not, and start_session now refuses that.
        nm0 = campaign.best_default()
        campaign.create(nm0)
        state = campaign.start_session(nm0)
    LIVE[sid] = state
    director = director_mod.Director(state)
    history: list = []
    auto = False                                     # hands-free (VAD) vs push-to-talk
    vad = VAD(sample_rate=config.SAMPLE_RATE, silence_ms=config.VAD_SILENCE_MS,
              min_speech_ms=config.VAD_MIN_SPEECH_MS, noise_mult=config.VAD_NOISE_MULT,
              ceiling=config.VAD_CEILING, barge_floor=config.VAD_BARGE_FLOOR)
    cast = voices_mod.VoiceCast.load(config.DM_CAMPAIGN)     # narrator + NPC voices
    book = speakers.SpeakerBook.load(config.DM_CAMPAIGN)     # learned player voices
    uplink = {"codec": "pcm16"}                       # phone says what it sends
    mic = {"frames": 0, "bytes": 0}                  # so "it can't hear me" has an answer
    rec = audiolog.SessionRecorder(getattr(state, "campaign", ""),
                                   getattr(state, "session", 0), sid,
                                   enabled=config.RECORD_AUDIO)
    enroll = {"target": None}                        # {"player","character","voice_id"} while enrolling
    # in-flight DM turn, run off the receive loop so the socket keeps draining. "flag" is
    # per-turn, not shared: an interrupted turn must stay silenced even after the next one
    # starts, or it would resume talking over it.
    turn = {"task": None, "playing": False, "flag": None}
    pending: set = set()                              # every turn task, for clean shutdown
    spoken_recently: list = []                        # the DM's last lines, to recognise its own echo
    combat_last: dict = {}                            # last initiative snapshot sent
    spec: dict = {"judged": False}                    # in-flight early transcription of this pause
    played = {"turns": 0}                             # did this connection actually play anything?
    notes = {"armed": False}                          # "feedback" said alone -> capture the next line
    amb_last = {"v": None}                            # last soundscape sent to this client
    director.on_event("session_start")
    logger.info(f"[{sid}] connected")
    room_mod.join(state.campaign, sock, sid)
    await sock.send_text(json.dumps({"type": "status", "msg": "connected"}))
    # Tell this phone who else is here and which characters are already taken, and tell the
    # others that somebody arrived.
    await room_mod.broadcast(state.campaign, room_mod.table(state.campaign))
    await sock.send_text(json.dumps({"type": "roster", "sid": sid,
                                     "characters": [r.get("character") for r in state.roster]}))
    # THE MAP SURVIVES A REFRESH. It only ever lived on the connection that drew it, so a reload
    # showed a blank screen while the table looked at a map that still existed on disk -- "maps
    # go away when you refresh", twice in one session.
    try:
        if svg := _render_map(state):
            await sock.send_text(json.dumps({"type": "map", "svg": svg, "quiet": True,
                                             "title": state.map_title or state.map_loc or "Map"}))
    except Exception as e:
        logger.debug(f"could not restore the map for a reconnecting phone: {e}")

    opened = {"done": False}

    async def open_session():
        """Fire the opening beat once: the DM sets the opening scene (new campaign) or recaps
        and drops the party back in (continuing). Runs on the explicit start button AND the
        first time the player enables the mic — so a brand-new campaign narrates itself the
        moment they're ready, instead of sitting silent until it's asked out loud."""
        if opened["done"]:
            return
        opened["done"] = True
        # ONCE PER SESSION. opened["done"] is per-connection, and phones reconnect constantly --
        # 181 sockets in one evening. Every reconnect re-opened the session, and the opener is
        # the single most expensive turn there is: 817 output tokens and ten seconds of silence
        # before the first word. The table asked for exactly this: "disable recap and continue
        # after the first time".
        if getattr(state, "opened_session", 0) == state.session:
            logger.info(f"[{sid}] session {state.session} already opened; not recapping again")
            return
        state.opened_session = state.session
        state.save()
        # judged from the campaign itself, exactly as the phone judged it when it labelled the
        # button -- not from whether this socket happened to resume.
        # A recap written at the END of the last session is already correct and already voiced,
        # so play it instead of spending fourteen seconds re-deriving it while the table sits in
        # silence. Only the recap: opening a NEW campaign is not a recap and has nothing stored.
        if (stored := recap_mod.load(state)) is not None:
            logger.info(f"[{sid}] serving the stored recap for session {stored.get('session')}")
            await sock.send_text(json.dumps({"type": "dm", "text": stored["text"]}))
            transcript.log(state.campaign, state.session, "dm", stored["text"], route="recap")
            if stored.get("wire") or stored.get("audio"):
                if rec is not None and stored.get("audio"):
                    rec.feed_far(audiolog.wav_to_pcm16k(stored["audio"]))
                # Already in wire form from build time; encoding it here would cost 2.5s and
                # hand back most of what precomputing the recap was for.
                await sock.send_bytes(stored.get("wire")
                                      or await tts.for_wire(stored["audio"]))
            else:                                    # text survived, audio did not: say it live
                for line in recap_mod.sentences(stored["text"]):
                    if a := await tts.synth(line):
                        # To the devices that are meant to make a noise -- one phone by the
                        # speakerphone -- not to whichever phone happened to hear the player.
                        await room_mod.play(state.campaign, await tts.for_wire(a), origin=sock)
            await sock.send_text(json.dumps({"type": "turn_end"}))
            if spoken_recently is not None:
                spoken_recently.append(stored["text"]); del spoken_recently[:-6]
            played["turns"] += 1
            await _send_combat(sock, state, combat_last)
            state.save()
            return
        opener = ("Open the campaign now." if campaign.is_unplayed(state)
                  else "Recap where we left off, then put us back in the scene.")
        played["turns"] += bool(await _handle_utterance(
            sock, sid, b"", state, director, history, cast, book,
            stop=None, on_accept=None, spoken=spoken_recently, rec=rec,
            speaking=lambda: turn["playing"],
            guess=_ready(opener), notes=None))
        await _send_combat(sock, state, combat_last)
        state.save()

    def new_flag() -> dict:
        turn["flag"] = {"stopped": False}
        return turn["flag"]

    async def start_turn(pcm: bytes, interrupts: bool, guess=None):
        """Begin a turn. If the DM is mid-answer this is a barge-in — but we don't cut it off
        until the audio has been transcribed and confirmed to be the player, since the loudest
        thing in the room right now is the DM itself."""
        prev = turn["flag"] if interrupts else None

        async def on_accept():
            if prev:
                prev["stopped"] = True               # confirmed player speech -> cut the DM off
                logger.info(f"[{sid}] barge-in")
                # Stop GENERATING it too, not just speaking it. The old turn used to run to
                # completion holding the backend lock while this one queued behind it, so the
                # player got silence and then a pile of stale answers.
                try:
                    backend_mod.get_backend().abandon()
                except Exception as e:
                    logger.debug(f"abandon failed: {e}")
                await sock.send_text(json.dumps({"type": "interrupted"}))

        flag = new_flag()
        task = asyncio.create_task(_run_turn(sock, finish, pcm, flag, on_accept, guess))
        turn["task"] = task
        pending.add(task); task.add_done_callback(pending.discard)
        await sock.send_text(json.dumps({"type": "turn_start"}))

    async def finish(pcm: bytes | None = None, flag: dict | None = None, on_accept=None,
                     guess=None):
        if pcm is None:
            pcm = bytes(buf)
        buf.clear()
        t = enroll["target"]
        if t:                                        # enrollment: learn this voice
            enroll["target"] = None
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            # The character name arrives as speech, so the voice book was labelled with what
            # the mic heard -- "Gleben", "Silvers Raily", "Araporno". add_player resolves it
            # for the roster; the book has to be given the same real name or a correct match
            # still announces the wrong character.
            if match := state.resolve(t["character"], pcs_only=True):
                t["character"] = match.name
            ok = book.enroll(t["voice_id"], arr, t["character"])
            state.add_player(t["player"], t["character"], voice_id=t["voice_id"]); state.save()
            await sock.send_text(json.dumps({"type": "status", "msg":
                f"✓ learned {t['player']}'s voice ({t['character']})" if ok else "didn't catch that — hold longer"}))
        else:
            played["turns"] += bool(await _handle_utterance(
                sock, sid, pcm, state, director, history, cast, book,
                stop=(lambda: flag["stopped"]) if flag else None,
                on_accept=on_accept, spoken=spoken_recently, guess=guess, notes=notes,
                speaking=lambda: turn["playing"],
                rec=rec))
            await _send_combat(sock, state, combat_last)
            if state.map_wanted:                 # the DM called for a battle map
                loc, state.map_wanted = state.map_wanted, ""
                t = asyncio.create_task(_draw_map(sock, state, loc, director))
                pending.add(t); t.add_done_callback(pending.discard)
            elif state.map_dirty:                # tokens moved or an area was revealed
                state.map_dirty = False
                if svg := _render_map(state):
                    # keep the library's fog in step, so returning here shows what they
                    # actually uncovered rather than resetting them to the entrance
                    maps.touch_seen(state.campaign, state.map_loc or state.map_title, state.map_seen)
                    await room_mod.broadcast(state.campaign, {"type": "map",
                        "title": state.map_title, "svg": svg, "quiet": True}, origin=sock)
            if ev := getattr(state, "director_event", None):
                # Opus thinks about what happens NEXT while Sonnet is still speaking this turn.
                state.director_event = None
                director.on_event(*ev)
            if not config.AMBIENCE:
                # Switched off, so the loop must never start -- and if one is already playing
                # from before it was turned off, say so once rather than leaving it running
                # under the narrator forever. One-shot [SFX] below are unaffected.
                if amb_last["v"] != "off":
                    amb_last["v"] = "off"
                    state.ambience = ""
                    await sock.send_text(json.dumps({"type": "ambience", "mood": "off",
                                                     "audio": None}))
            elif state.ambience and state.ambience != amb_last["v"]:  # mood changed -> new soundscape
                amb_last["v"] = state.ambience
                # "off" means silence, not a mood: a null payload pauses the loop. Without
                # this the generator's catch-all renders a neutral drone for the word "off".
                if state.ambience.strip().lower() in ("off", "none", "silence", "stop", "quiet"):
                    await sock.send_text(json.dumps({"type": "ambience", "mood": "off", "audio": None}))
                else:
                    url = "data:audio/wav;base64," + base64.b64encode(ambience.wav_bytes(state.ambience)).decode()
                    await sock.send_text(json.dumps({"type": "ambience", "mood": state.ambience, "audio": url}))
                logger.info(f"ambience -> {state.ambience}")
            while state.sfx_queue:               # dramatic one-shots, over the top of the loop
                name = state.sfx_queue.pop(0)
                url = "data:audio/wav;base64," + base64.b64encode(ambience.sfx_bytes(name)).decode()
                await sock.send_text(json.dumps({"type": "sfx", "name": name, "audio": url}))
                logger.info(f"sfx -> {name}")
            state.save()

    try:
        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (b := msg.get("bytes")) is not None:
                if uplink["codec"] == "ulaw":
                    b = stt.ulaw_to_pcm16(b)     # everything downstream still sees 16-bit PCM
                # "it isn't hearing me" is three different faults that look identical from the
                # phone: no frames at all, frames arriving with hands-free off (so they pile up
                # in buf and are never looked at), or frames that reach the VAD and never pass
                # it. Say which, once per connection, instead of leaving it to be guessed.
                mic["frames"] += 1
                mic["bytes"] += len(b)
                rec.feed(b)                  # keep the raw audio; decisions get indexed to it
                if mic["frames"] in (1, 200):
                    level = float(np.abs(np.frombuffer(b, dtype="<i2").astype(np.float32)
                                         / 32768.0).mean()) if len(b) >= 2 else 0.0
                    logger.info(f"[{sid}] audio in: frame {mic['frames']}, "
                                f"{mic['bytes']:,} bytes total, auto={auto}, "
                                f"level={level:.4f}"
                                + ("" if auto else "  <-- hands-free OFF: these frames are "
                                                   "buffered for push-to-talk and never "
                                                   "reach the VAD"))
                if not auto or enroll["target"] is not None:
                    buf.extend(b)                # push-to-talk / enrollment: client marks the span
                    continue
                # ALWAYS LISTENING. The mic is never gated, so there is no re-arm delay after
                # the DM speaks -- the player can start talking the instant it stops, or over
                # the top of it. The VAD just gets stricter while the DM holds the floor.
                busy = _busy(turn)
                vad.set_speaking(busy)
                if (utt := vad.feed(b)) is not None:
                    # `interrupts` is what cuts the DM off mid-sentence. At a table, people talk
                    # over narration constantly and mean nothing by it -- see config.ALLOW_BARGE_IN.
                    await start_turn(utt, interrupts=busy and config.ALLOW_BARGE_IN,
                                     guess=spec.pop("task", None))
                elif (early := vad.speculate()) is not None:
                    # transcribe DURING the end-of-turn pause, not after it
                    if (old := spec.pop("task", None)) is not None:
                        old.cancel()                 # they kept talking; that guess is stale
                    spec["task"] = asyncio.create_task(
                        asyncio.to_thread(stt.transcribe, _to_f32(early), state.vocabulary()))
                    spec["judged"] = False
                elif spec.get("task") and not spec["judged"] and spec["task"].done():
                    # The words are in before the pause is up, so decide on them rather than
                    # on silence alone: answer a finished thought at once, wait longer on a
                    # half-sentence, and stay out of table talk entirely.
                    spec["judged"] = True
                    heard = (spec["task"].result() or "").strip()
                    # speculative=True: this transcript is of audio we already know is
                    # unfinished, so its trailing full stop is Whisper's, not the player's.
                    call, why = endpoint.assess(heard, _party(state), busy, speculative=True)
                    # In a room with a table in it, most of what the microphone hears is
                    # people talking to each other. Answering that is what made the DM say
                    # "Silence." four times running at people discussing someone's dad.
                    if (call == endpoint.GO and _group(state)
                            and not endpoint.for_the_dm(heard, _party(state), state.in_combat)):
                        call, why = endpoint.IGNORE, "table talk, not for the DM"
                    if (notes["armed"] or feedback.parse(heard) is not None
                            or safety.is_x_card(heard)):
                        # Feedback is spoken in longer, more considered stretches than play,
                        # and cutting it in half loses the point being made. Never fast-path
                        # it; give them room to finish the thought.
                        call, why = endpoint.WAIT, "feedback — let them finish"
                    if call == endpoint.GO and (utt := vad.finish_now()) is not None:
                        logger.info(f"[{sid}] answering early ({why}): {heard!r}")
                        rec.mark("answered", text=heard, why=why, speaking=busy)
                        await start_turn(utt, interrupts=busy, guess=spec.pop("task", None))
                    elif call == endpoint.WAIT:
                        vad.hold(config.VAD_FEEDBACK_HOLD_MS if "feedback" in why
                                 else config.VAD_HOLD_MS)
                        logger.info(f"[{sid}] holding ({why}): {heard!r}")
                        rec.mark("held", text=heard, why=why, speaking=busy)
                    elif call == endpoint.IGNORE:
                        logger.info(f"[{sid}] not for me ({why}): {heard!r}")
                        rec.mark("ignored", text=heard, why=why, speaking=busy)
                        vad.reset(); spec.pop("task", None)
                continue
            data = json.loads(msg.get("text") or "{}")
            if data.get("type") == "uplink":
                uplink["codec"] = "ulaw" if data.get("codec") == "ulaw" else "pcm16"
                logger.info(f"[{sid}] uplink codec: {uplink['codec']}")
            elif data.get("type") == "start":
                buf.clear()
            elif data.get("type") == "end":
                pcm = bytes(buf); buf.clear()
                await start_turn(pcm, interrupts=_busy(turn))
            elif data.get("type") == "enroll":       # next utterance teaches this player's voice
                enroll["target"] = {"player": data.get("player", "Player"),
                                    "character": data.get("character", data.get("player", "Hero")),
                                    "voice_id": data.get("player", "p").strip().lower()}
                buf.clear()
                await sock.send_text(json.dumps({"type": "status",
                    "msg": f"say a sentence as {enroll['target']['character']}…"}))
            elif data.get("type") == "listen":       # start hands-free auto mode
                auto = True; vad.reset(); buf.clear()
                logger.info(f"[{sid}] hands-free ON (listen)")
                await sock.send_text(json.dumps({"type": "status", "msg": "listening"}))
                # Enabling the mic is a user gesture, so audio is now unlocked. A brand-new
                # campaign should open itself here rather than wait to be asked out loud; a
                # session already in progress keeps its explicit "recap and continue" button.
                if not state.resumed and campaign.is_unplayed(state):
                    await open_session()
            elif data.get("type") == "seat":
                # This phone says who it is playing. Deterministic attribution: a tap beats a
                # similarity score, and at a real table the enrolled voice profiles sat up to
                # 0.703 similar to each other and correctly refused to name anyone.
                tbl = room_mod.sit(state.campaign, sock, data.get("character") or "",
                                   getattr(state, "roster", []))
                await room_mod.broadcast(state.campaign, tbl)
            elif data.get("type") == "audio":
                # Which devices make a noise. One phone by the speakerphone plays the DM; the
                # rest show the map and stay quiet, or seven of them narrate at once.
                tbl = room_mod.set_audio(state.campaign, sock, bool(data.get("on", True)))
                await room_mod.broadcast(state.campaign, tbl)
            elif data.get("type") == "interrupt":     # explicit tap to cut the DM off
                if turn["flag"]:
                    turn["flag"]["stopped"] = True    # stop speaking; the task drains quietly
                turn["playing"] = False
                logger.info(f"[{sid}] interrupted by player")
                await sock.send_text(json.dumps({"type": "interrupted"}))
            elif data.get("type") == "begin":
                # The explicit start button. Same beat the mic-enable auto-fires, guarded so
                # tapping Begin after it already opened doesn't play the intro twice.
                await open_session()
            elif data.get("type") == "text":
                # A typed turn: the same path a spoken one takes, minus the microphone. Used
                # by tools/playtest.py to actually play the game and check what comes back --
                # every fault of the last week was findable this way and none of them were
                # found this way, because nothing could take a turn without a person.
                said = (data.get("text") or "").strip()
                if said:
                    logger.info(f"[{sid}] typed: {said!r}")
                    if _busy(turn):                  # supersede, exactly as speaking would
                        if turn["flag"]:
                            turn["flag"]["stopped"] = True
                        try:
                            backend_mod.get_backend().abandon()
                        except Exception:
                            pass
                    turn["flag"] = new_flag()
                    turn["task"] = asyncio.create_task(_handle_utterance(
                        sock, sid, b"", state, director, history, cast=cast, book=book,
                        stop=lambda f=turn["flag"]: bool(f.get("stopped")),
                        spoken=spoken_recently, guess=_ready(said), notes=notes, rec=rec,
                        speaking=lambda: turn["playing"]))
                    pending.add(turn["task"])
                    turn["task"].add_done_callback(pending.discard)
                    played["turns"] += 1
            elif data.get("type") == "playing":       # client's speaker state, for barge-in bias
                turn["playing"] = bool(data.get("on"))
            elif data.get("type") == "stop_listen":
                logger.info(f"[{sid}] hands-free OFF (stop_listen)")
                auto = False
                turn["playing"] = False
            elif data.get("type") == "campaign":     # switch to another campaign
                nm = (data.get("name") or config.DM_CAMPAIGN).strip()
                if nm != state.campaign:
                    # Only recap-and-save the outgoing campaign when actually leaving it.
                    # Re-requesting the SAME campaign (e.g. reloading right after a history
                    # reset) must not resave this stale in-memory state, or it clobbers the
                    # just-reset file on disk with the pre-reset session/recap/location.
                    try:
                        await planner.recap_session(state); state.save()
                    except Exception:
                        pass
                try:
                    state = campaign.start_session(nm)
                except FileNotFoundError as e:
                    await sock.send_text(json.dumps({"type": "status", "msg": f"no campaign '{nm}'"}))
                    logger.warning(f"[{sid}] {e}")
                    continue
                LIVE[sid] = state
                director = director_mod.Director(state)
                history = []
                cast = voices_mod.VoiceCast.load(nm)
                book = speakers.SpeakerBook.load(nm)
                amb_last["v"] = None
                opened["done"] = False           # the newly-selected campaign may open itself
                director.on_event("session_start")
                await sock.send_text(json.dumps({"type": "status",
                    "msg": f"▶ {nm} — session {state.session}"}))
    except WebSocketDisconnect:
        pass
    finally:
        rec.close()
        LIVE.pop(sid, None)
        for t in list(pending):
            t.cancel()
        # Only persist a session that actually happened. Otherwise merely opening the page
        # would resurrect a campaign the player just deleted, and write an empty recap.
        if played["turns"]:
            try:
                await planner.recap_session(state)    # remember this session for next time
                state.save()
            except Exception as e:
                logger.warning(f"recap/save failed: {e}")
        else:
            logger.info(f"[{sid}] no turns played — nothing saved")
        room_mod.leave(state.campaign, sock)
        await room_mod.broadcast(state.campaign, room_mod.table(state.campaign))
        logger.info(f"[{sid}] disconnected")


def _busy(turn) -> bool:
    """Does the DM currently have the floor — still generating, or still being played out?"""
    t = turn["task"]
    return (t is not None and not t.done()) or turn["playing"]


def _flush_pending(pending: dict, queue, synth) -> None:
    """Turn the buffered sentences into ONE synthesis call.

    Merging is what avoids paying the ~4.2s fixed cost per sentence; the joined text is spoken
    as a single utterance, which also reads better than several clipped ones stitched together.
    """
    if not pending["parts"]:
        return
    line = " ".join(pending["parts"]).strip()
    voice = pending["voice"]
    pending["parts"] = []
    pending["voice"] = None
    if line:
        queue.put_nowait(asyncio.create_task(synth(line, voice)))


async def _run_turn(sock, finish, pcm: bytes, flag: dict, on_accept=None, guess=None):
    """A turn, run off the receive loop so the socket keeps draining while the DM thinks and
    talks. A failure here must not kill the session."""
    try:
        await finish(pcm, flag, on_accept, guess)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"turn failed: {e}")
        try:
            await sock.send_text(json.dumps({"type": "status", "msg": "(that turn failed)"}))
        except Exception:
            pass
    finally:
        # Always announced, including the "didn't catch that" and failure paths: the client
        # waits for this to know the turn is over and to report when playback finishes.
        try:
            await sock.send_text(json.dumps({"type": "turn_end"}))
        except Exception:
            pass


async def _one(line: str):
    yield line


def _ready(text: str):
    """A already-known 'transcript', so the start button reuses the normal turn path instead
    of duplicating it."""
    async def done():
        return text
    return asyncio.ensure_future(done())


def _render_map(state, reveal_all=False) -> str:
    """Redraw the current map with everyone's latest positions and whatever they've explored.

    Geometry first. battlemap lays a place out as polygons from what KIND of place it is -- a
    quay always has water, a vault always has shelving -- and mapart draws it as vector art: a
    curved shoreline instead of a staircase of squares, timber with planks, buildings with a
    cast shadow. The ASCII grid was the ceiling on the old maps, not the content: one character
    per five-foot square cannot express a diagonal, so no amount of better description helped.

    14ms for seven maps, no model call, and the same place every time the party returns.
    """
    loc = getattr(state, "map_loc", "") or getattr(state, "location", "") or state.map_title
    if loc:
        try:
            import battlemap
            import mapart
            scene = battlemap.build(loc, title=state.map_title or loc)
            toks = []
            for c in state.combatants.values():
                if c.hp <= 0 and not c.is_pc:
                    continue
                if getattr(c, "pos", None) and len(c.pos) >= 2:
                    toks.append((c.pos[0] + 0.5, c.pos[1] + 0.5, c.name[:2], c.is_pc))
            state.map_svg = mapart.render(scene, tokens=toks, note=f"{scene.kind} · 5 ft squares")
            return state.map_svg
        except Exception as e:
            logger.warning(f"vector map failed for {loc!r} ({e}); falling back to the grid")
    if not state.map_grid:
        return ""
    rects = [tuple(r[:4]) for r in state.map_rooms]
    tokens = planner.place_tokens(state.map_grid, rects, state)
    visible = None if reveal_all else planner.visible_cells(state.map_grid, rects, state.map_seen)
    # label at the top of each room rather than the middle, so a token never sits on the name
    rooms = [(r[0] + r[2] // 2, r[1], r[4]) for r in state.map_rooms if len(r) > 4]
    if visible is not None:                              # don't name rooms they haven't seen
        rooms = [(x, y, n) for x, y, n in rooms if (x, y) in visible]
    # an outdoor site must not be labelled and coloured as dungeon furniture
    kind = planner.map_kind(state.location or state.map_title or "")
    state.map_svg = planner.grid_to_svg(state.map_grid, title=state.map_title, rooms=rooms,
                                        tokens=tokens, visible=visible, kind=kind)
    return state.map_svg


def _prepped_names(director, location: str, n: int):
    """Room names the cartographer worked out ahead of time, if this location was prepped."""
    raw = (getattr(director, "map_prep", {}) or {}).get((location or "").strip().lower(), "")
    if not raw:
        return None, ""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    feature = next((l.split(":", 1)[1].strip() for l in lines if l.upper().startswith("FEATURE:")), "")
    names = [l for l in lines if not l.upper().startswith("FEATURE:")][:n]
    return (names or None), feature


async def _draw_map(sock, state, location: str, director=None):
    """Generate and deliver a battle map off the play loop. Takes seconds, so the DM keeps
    talking and the map lands on their screen when it's ready."""
    try:
        title = planner.map_title(location)
        # A place already mapped is restored exactly as it was -- same layout, same room
        # names, same fog -- instead of being reinvented as a different place each visit.
        if saved := maps.load(state.campaign, location):
            state.map_grid = saved["grid"]
            state.map_rooms = saved["rooms"]
            state.map_seen = saved.get("seen") or [0]
            state.map_title = saved.get("title") or title
            state.map_loc = location
            svg = _render_map(state)
            await room_mod.broadcast(state.campaign, {"type": "map",
                "title": state.map_title, "svg": svg, "restored": True}, origin=sock)
            n = maps.visit(state.campaign, location)
            logger.info(f"map restored from library: {state.map_title} (visit {n})")
            return
        await sock.send_text(json.dumps({"type": "status", "msg": f"drawing {title}…"}))
        # Draw the place the fiction describes, if we can. The procedural generator is seeded
        # by the location STRING and cannot know anything about where they are -- it drew a
        # dungeon of rooms and corridors for a stone quay. cartograph reads the prepared
        # briefing (or the live scene, for somewhere nobody planned) and lays out what is
        # actually there; it returns nothing at all rather than a map it could not verify.
        import cartograph
        drawn, named = await cartograph.draw(state, location, title)
        if drawn:
            grid = drawn
            rects = planner.rooms_of(grid) or [(1, 1, 22, 14)]
        else:
            grid, rects = planner.make_map_full(location)  # instant, deterministic
        centres = [planner.room_centre(r) for r in rects]
        prepped, feature = _prepped_names(director, location, len(rects))
        if drawn and named:
            # Names straight out of the description -- "Wren's loft", "Dorrit's stall" -- beat
            # anything a model invents for a room it has not been told about.
            prepped = [n[2] for n in named][:len(rects)] or prepped
        if prepped:
            # Opus named these between turns, so the map lands immediately instead of waiting
            # on a model at the exact moment a fight is starting.
            names = [(x, y, nm) for (x, y), nm in zip(centres, prepped)]
            logger.info(f"map: using cartographer's prepped names for {location!r}")
        else:
            names = await planner.name_rooms(title, centres)
        if feature:
            state.remember(f"Battle map of {title}: {feature}")
        state.map_grid = grid
        state.map_rooms = [[*r, n] for r, (_, _, n) in zip(rects, names)] or \
                          [[*r, f"Area {i+1}"] for i, r in enumerate(rects)]
        state.map_seen = [0]                             # only where they're standing
        state.map_title = title
        state.map_loc = location
        svg = _render_map(state)
        maps.save(state.campaign, location, title=title, grid=state.map_grid,
                  rooms=state.map_rooms, seen=state.map_seen)
        # Everyone sees the map, not just whoever happened to walk into the room.
        await room_mod.broadcast(state.campaign, {"type": "map", "title": title, "svg": svg}, origin=sock)
        logger.info(f"map drawn + saved: {title} ({len(centres)} rooms)")
    except Exception as e:
        logger.warning(f"map failed ({location}): {e}")
        try:
            await sock.send_text(json.dumps({"type": "status", "msg": "couldn't draw that map"}))
        except Exception:
            pass


async def _tell(sock, payload: dict) -> None:
    """Send to a phone that is not the one driving this turn; never let it break the turn.

    These sockets belong to other players and can vanish at any moment -- a locked phone, a
    dropped tunnel. A write failing to one of them must not take down the turn that is
    currently speaking for the whole table.
    """
    try:
        await sock.send_text(json.dumps(payload))
    except Exception:
        pass


async def _say(sock, line: str, campaign: str = ""):
    """A short spoken acknowledgement that isn't part of the story.

    `campaign` is how the voice reaches the device holding the audio rather than whichever
    phone happened to be listening. It is optional and falls back to this socket, because the
    alternative was what actually shipped: a bulk edit put `state.campaign` in here, _say has
    no `state`, and every acknowledgement it makes -- "Noted.", the safety replies, and the new
    whose-turn-is-it prompt -- raised NameError into a silent except for a whole session.
    """
    await sock.send_text(json.dumps({"type": "dm", "text": line}))
    if audio := await tts.synth(line):
        wire = await tts.for_wire(audio)
        if campaign:
            await room_mod.play(campaign, wire, origin=sock)
        else:
            try:
                await sock.send_bytes(wire)
            except Exception:
                pass


async def _send_combat(sock, state, last: dict):
    """Initiative, to every phone at the table. Combat is unreadable by voice alone and the
    player who happened to speak is not the only one who needs to see whose turn it is."""
    """Push the initiative order whenever it changes. Combat is unreadable by voice alone —
    who's up, who's next, what round — and that is most of what 'figure out initiative' means."""
    snap = {
        "in_combat": state.in_combat,
        "round": state.round,
        "turn": state.current_turn() or "",
        "pending": list(state.pending_init),
        # HOW MUCH HAVE WE PUT INTO IT. Asked for outright: "can enemy we attack show the
        # amount of damage we currently have done". The engine has always known a monster's
        # exact hit points and deliberately never says them out loud -- a DM reading HP aloud
        # ruins the guesswork -- but a bar on the screen is not the same thing as reading it
        # out, and the table cannot judge whether to swing again or run without it.
        "order": [{"name": n, "init": (c.initiative if (c := state.get(n)) else None),
                   "pc": bool(c and c.is_pc),
                   "hurt": (round(100 * (c.max_hp - c.hp) / c.max_hp)
                            if c and not c.is_pc and c.max_hp else None),
                   "dealt": ((c.max_hp - c.hp) if c and not c.is_pc else None),
                   "state": (c.status if c and c.is_pc else (c.vibe if c else ""))}
                  for n in state.order],
    }
    if snap == last.get("v"):
        return
    last["v"] = snap
    await room_mod.broadcast(getattr(state, "campaign", ""), {"type": "combat", **snap},
                             origin=sock)


def _group(state) -> bool:
    """More than one player character means the microphone is hearing a ROOM.

    Solo, almost everything said is meant for the DM and gating it would only get in the way.
    With a table of seven, the reverse is true.
    """
    return sum(1 for c in state.combatants.values() if c.is_pc) > 1


def _party(state) -> list:
    """Names at the table, so speech aimed at another player isn't answered by the DM."""
    out = []
    for r in (state.roster or []):
        out += [r.get("character") or "", r.get("player") or ""]
    return [n for n in out if n]


def _to_f32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _words(s: str) -> set:
    # Possessives and contractions have to collapse to the root or the DM's own line fails to
    # match itself: it said "the water's still pulling at you" and heard back "the water is
    # still pulling at you", and water's != water was enough to let the echo through.
    s = re.sub(r"'(s|re|ve|ll|d|m|t)\b", "", s.lower())
    return {w for w in re.findall(r"[a-z]+", s) if len(w) > 3}


def _is_echo(text: str, spoken: list) -> bool:
    """Is this the DM's own voice coming back through the phone speaker?

    Two lengths matter and they are not the same one. The utterance must be a real sentence
    (raw words) before it is worth judging at all, and it must carry enough CONTENT words for
    the overlap to mean anything. Measuring both on the filtered set was the bug: "The water
    is still pulling at you" is seven words but only three content words, so it was written
    off as too short to judge and the DM interrupted itself with its own narration.

    The other direction still has to hold. "Is it my turn now?" is a real sentence carrying a
    single content word — "turn" — which the DM says constantly in combat, and scoring that as
    a perfect match silently ignores the player. Wrongly dropping a real question looks like
    the app is dead; wrongly answering a scrap of echo costs one stray turn. So: long enough
    to be a sentence, rich enough to compare, and then a high bar on the overlap.
    """
    raw = re.findall(r"[a-z']+", (text or "").lower())
    heard = _words(text)
    # These stay deliberately loose, because a real echo can be short: "The water is still
    # pulling at you" carries three content words and must still be caught. The reason that no
    # longer swallows the player is that this is only ever consulted WHILE THE DM IS SPEAKING
    # -- see the call site. Text alone cannot tell "the water is still pulling at you" coming
    # back off a speaker from a player asking about the water, and it should not try.
    if len(raw) < 5 or len(heard) < 3:
        return False
    said = _words(" ".join(spoken[-4:]))
    if not said:
        return False
    return len(heard & said) / len(heard) >= 0.7


async def _for_the_dm(text: str, party, in_combat: bool) -> bool:
    """Should the story model see this at all? Three steps, cheapest first.

    Measured over a real seven-player evening: 417 turns went to the metered model and 126 of
    them produced no audio, because the room was talking to itself and the model correctly
    declined to answer. Each of those cost a call, held the turn desk while real play queued,
    and looked to the table like a DM thinking and then saying nothing. "Way too slow" was
    mostly this -- the turns that DID answer took a median of 2.5 seconds.
    """
    import addressed
    if endpoint.certainly_for_dm(text, party, in_combat):
        return True                                  # plainly play: no model, no wait
    if not endpoint.for_the_dm(text, party, in_combat):
        return False                                 # plainly said to somebody else
    # Genuinely ambiguous, which is most of what a room says. ~47ms on the local model, free,
    # and it answers YES whenever it is unsure or unavailable.
    return await addressed.ask(text, party)


async def _handle_utterance(sock, sid, pcm_bytes, state, director, history, cast=None, book=None,
                            stop=None, on_accept=None, spoken=None, guess=None,
                            notes=None, rec=None, speaking=None) -> bool:
    """Returns True if a real turn was played (so the caller knows the session is worth saving)."""
    if len(pcm_bytes) < 2 and guess is None:
        return False                                 # no audio and nothing already transcribed
    pcm = _to_f32(pcm_bytes)

    t0 = time.perf_counter()
    if guess is not None:                            # transcription already ran during the pause
        text = await guess
    else:
        text = await asyncio.to_thread(stt.transcribe, pcm, state.vocabulary())
    t_stt = (time.perf_counter() - t0) * 1000
    if not text:
        return False                                 # silence/noise: say nothing, keep listening
    # Echo is the DM's own voice coming back out of the room's speaker, so it can only happen
    # while the DM is ACTUALLY SPEAKING. The check used to run on every utterance against the
    # last four things said, long after the audio had stopped -- so a player asking about the
    # scene they had just been told about was discarded as an echo of it, and the app looked
    # deaf. If nothing is playing, whatever was heard is the player.
    if spoken and (speaking is None or speaking()) and _is_echo(text, spoken):
        logger.info(f"[{sid}] ignored own echo: {text!r}")
        if rec is not None:
            rec.mark("echo", text=text)
        return False
    if on_accept:                                    # real speech -> now it's safe to cut the DM off
        await on_accept()

    # Feedback is intercepted before anything else: it must not reach the DM, take a turn,
    # or touch the campaign. Answered in one word so it barely interrupts play.
    if notes is not None:
        note = feedback.parse(text)
        if notes.get("armed") and note is None:      # the note is whatever they said next
            notes["armed"] = False
            note = text
        if note is not None:
            await sock.send_text(json.dumps({"type": "transcript", "text": text,
                                             "speaker": None, "ms": round(t_stt)}))
            if not note:                             # just the trigger word -> take the next line
                notes["armed"] = True
                await _say(sock, "Go ahead.", state.campaign)
                await sock.send_text(json.dumps({"type": "status", "msg": "📝 listening for feedback"}))
                return False
            item = feedback.add(note, state.campaign, state.session)
            logger.info(f"[{sid}] FEEDBACK: {note!r}")
            await sock.send_text(json.dumps({"type": "feedback", "item": item}))
            await _say(sock, "Noted.", state.campaign)
            return False

    # Safety tools sit right behind feedback: before the DM sees anything, before speaker ID.
    if safety.is_x_card(text):
        logger.info(f"[{sid}] X-CARD")
        await sock.send_text(json.dumps({"type": "transcript", "text": text, "speaker": None,
                                         "ms": round(t_stt)}))
        await sock.send_text(json.dumps({"type": "status", "msg": "🛑 moving on"}))
        text = player_text = safety.PULL_AWAY          # the DM gets the directive, not the word
    elif (sv := safety.parse_set(text)):
        await _say(sock, safety.add(state, *sv), state.campaign)
        return False
    elif safety.is_list_request(text):
        await _say(sock, safety.describe(state), state.campaign)
        return False

    # WHO SPOKE. A player who tapped their own character on their own phone is a fact; a
    # voice embedding is a similarity score, and at a real seven-player table the enrolled
    # profiles sat up to 0.703 similar to one another and correctly refused to name anybody.
    # The seat wins whenever there is one, and the voice model stays for the shared-phone case.
    speaker = room_mod.speaking_for(getattr(state, "campaign", ""), sock) or None
    if speaker is None and book is not None and pcm.size > config.SAMPLE_RATE // 4:
        _, speaker, _ = book.identify(pcm)
    if text is not safety.PULL_AWAY:
        player_text = f"[{speaker}] {text}" if speaker else text
    # EVERY PHONE, not just the one that heard it. The DM's words were already broadcast and
    # the players' were not, so each person saw only their own half of the conversation --
    # which at a table of seven reads as the app missing most of what was said.
    await room_mod.broadcast(state.campaign, {"type": "transcript", "text": text,
                                              "speaker": speaker, "ms": round(t_stt)},
                             origin=sock)
    logger.info(f"[{sid}] heard{f' ({speaker})' if speaker else ''}: {text!r} (stt {t_stt:.0f}ms)")

    transcript.log(state.campaign, state.session, "player", text,
                   speaker=speaker, stt_ms=round(t_stt))

    if cast is not None:                             # who's on stage -> resolves unquoted dialogue
        cast.known = {c.name.lower() for c in state.combatants.values()} | \
                     {r["character"].lower() for r in state.roster}

    # Speech is PIPELINED: each sentence is sent to the voice engine the moment the DM writes
    # it, while earlier ones are still playing. Synthesis is a ~0.5-2s network round trip, so
    # doing it strictly in lockstep left audible dead air between every sentence.
    t_first_token = t_first_audio = None
    desk, have_floor, mirrors = turns.desk(state.campaign), False, []
    queue: asyncio.Queue = asyncio.Queue()
    gate = asyncio.Semaphore(3)                      # don't open a dozen TTS connections at once

    async def synth(line, voice):
        async with gate:
            return await tts.synth(line, voice)

    # Field reads and combat bookkeeping are answered in Python with no model involved: it is
    # ~0.1us instead of ~500ms, and a number read off the sheet cannot be hallucinated.
    # quick.answer() returns None unless it is certain, in which case a model handles it.
    # Note any spell or weapon just named, on either side, so "what dice for THIS" resolves
    # locally instead of costing a model round trip.
    # THE TABLE MOVED ON. A different player declaring an action while somebody else's turn is
    # still open is a fact, not a guess about intent -- and without it the fight waits forever
    # on a player who has put their phone down, which is what kept a real session in round 1.
    if state.in_combat and speaker and config.AUTO_HANDOVER:
        import combat as _cb
        if _cb.took_over(state, speaker):
            held = _cb.waiting_on_pc(state)
            logger.info(f"[{sid}] {speaker} acted while {held} was up; handing the turn over")
            state.pc_acted = False
            state.asked_end = False
            state.end_nudges = 0
            state.next_turn()
    quick.remember_subject(text, state)
    if (fixed := quick.answer(text, state, speaker)) is not None:
        logger.info(f"[{sid}] answered in python: {fixed!r}")
        source, route = _one(fixed), "python"
    elif _group(state) and not await _for_the_dm(text, _party(state), state.in_combat):
        # The same gate as the speculative path, applied HERE too. That one only judges
        # utterances the endpoint decides to answer EARLY; anything a player simply finished
        # saying went straight through it, so half the room's conversation still reached the
        # DM. One rule, every entry point.
        logger.info(f"[{sid}] not for me (table talk): {text!r}")
        if rec is not None:
            rec.mark("ignored", text=text, why="table talk")
        return False
    else:
        # THE DESK. Four phones are four independent callers into one DM, and nothing used to
        # coordinate them -- the turns did not overlap, they queued, each waiting out every
        # turn in front of it. Three players speaking inside twenty seconds is what every
        # 60-second turn in the logs looks like. Wait for the floor, then answer everyone who
        # spoke while it was busy in this one turn: three players cost one generation instead
        # of three, and nobody waits longer than a single turn.
        ticket = desk.hold(speaker, text, sock)
        if desk.busy:
            # Told, not awaited. Awaiting anything between holding a ticket and asking for the
            # floor hands the lock queue to whoever skipped the await -- the player who spoke
            # FIRST was answered LAST, purely because they were the one shown the notice.
            asyncio.create_task(sock.send_text(json.dumps(
                {"type": "status", "msg": "… the DM is mid-sentence"})))
        held = await desk.floor(ticket)
        if held is None:
            # Somebody else's turn already answered this line. Saying it again would be the
            # DM repeating itself at the table.
            logger.info(f"[{sid}] folded into another player's turn: {text!r}")
            return False
        have_floor = True
        if len(held) > 1:
            player_text = turns.merge(held)
            logger.info(f"[{sid}] answering {len(held)} players together: "
                        + " | ".join(repr(t.text[:40]) for t in held))
            # The other players' phones are about to go quiet while this socket does the
            # talking for all of them. Mirror the words to their screens, or being answered
            # looks exactly like being ignored. Text only: the room needs ONE voice, and the
            # audio belongs to whichever device is next to the speaker.
            mirrors = [t.sock for t in held if t.sock is not None and t.sock is not sock]
            for m in mirrors:
                asyncio.create_task(_tell(m, {"type": "status",
                                              "msg": "… answered together with the others"}))
        source = llm.stream_dm(player_text, state, director, history)
        route = "ollama" if backend_mod.is_quick_question(text) else config.DM_BACKEND

    pending = {"parts": [], "voice": None, "sent_first": False}

    async def produce():
        nonlocal t_first_token
        try:
            async for sentence in source:
                if t_first_token is None:
                    t_first_token = (time.perf_counter() - t0) * 1000
                # Barge-in: stop SPEAKING at once, but keep consuming the generator. The Claude
                # Code session is one long-lived process, so abandoning it mid-response would
                # leave the rest of this turn in the pipe to surface inside the next one.
                if stop is not None and stop():
                    continue
                # Every phone shows the words, whether or not it is making the sound.
                await room_mod.broadcast(state.campaign, {"type": "dm", "text": sentence}, origin=sock)
                quick.remember_subject(sentence, state)
                transcript.log(state.campaign, state.session, "dm", sentence, route=route)
                if spoken is not None:
                    spoken.append(sentence); del spoken[:-6]
                voice, line = cast.line_voice(sentence) if cast else (None, sentence)
                # Synthesis costs ~4.2s per CALL plus ~0.03s per character, measured: 15 chars
                # takes 4.63s and 163 takes 8.96s. One task per sentence therefore pays that
                # fixed cost once per sentence, and a three-sentence answer burns ~12s of pure
                # overhead. But the first chunk must still be short, because nothing is heard
                # until it lands. So: speak sentence one immediately, then coalesce.
                if not pending["sent_first"]:
                    pending["sent_first"] = True
                    queue.put_nowait(asyncio.create_task(synth(line, voice)))
                    continue
                # only consecutive lines in the SAME voice may merge, or an NPC's line would
                # come out in the narrator's
                if pending["voice"] is not None and voice != pending["voice"]:
                    _flush_pending(pending, queue, synth)
                pending["voice"] = voice
                pending["parts"].append(line)
                if sum(len(p) for p in pending["parts"]) >= config.TTS_BATCH_CHARS:
                    _flush_pending(pending, queue, synth)
        finally:
            _flush_pending(pending, queue, synth)
            queue.put_nowait(None)

    # From here the floor is HELD, and the only thing worse than a slow turn is a desk that is
    # never given back: every later player would wait forever on a turn that has already ended.
    # The finally below covers the loop; this covers the few statements before it.
    try:
        producer = asyncio.create_task(produce())
    except BaseException:
        if have_floor:
            desk.release()
        raise
    gone = False
    try:
        while (task := await queue.get()) is not None:
            if gone or (stop is not None and stop()):
                task.cancel()
                continue
            audio = await task
            if audio:
                if t_first_audio is None:
                    t_first_audio = (time.perf_counter() - t0) * 1000
                try:
                    # keep what went to the speaker, aligned with the mic, so echo
                    # cancellation has a reference to subtract
                    if rec is not None:
                        rec.feed_far(audiolog.wav_to_pcm16k(audio))
                    # To the devices that are meant to make a noise -- one phone by the
                    # speakerphone -- not to whichever phone happened to hear the player.
                    await room_mod.play(state.campaign, await tts.for_wire(audio), origin=sock)
                except (RuntimeError, WebSocketDisconnect) as e:
                    # The phone went away mid-narration — locked, backgrounded, or the tunnel
                    # blinked. Writing into a closed socket raises out of the whole handler,
                    # which killed the turn AND the connection, so an intro that dropped at
                    # sentence four was lost entirely and nothing was ever saved.
                    logger.info(f"[{sid}] client left mid-turn ({type(e).__name__}); "
                                "draining quietly")
                    gone = True
    finally:
        await producer
        # The floor is given up here and nowhere else. A turn that raises, disconnects or is
        # cut off still has to release it, or the desk stays locked and every later player
        # waits on a turn that ended.
        if have_floor:
            desk.release()
    if gone:
        return True                                  # it was a real turn; let the caller save it

    # A turn that says nothing at all is never the story being quiet -- it means the brain
    # failed. Silence with no explanation is the worst version of that, so say what happened.
    #
    # Unless WE cut it off. A turn the player superseded by speaking again said nothing on
    # purpose, and warning "the DM didn't answer that one" for it is both wrong and alarming:
    # a playtest that spoke over a slow turn twice got two warnings for working correctly.
    if t_first_audio is None and not gone and not (stop is not None and stop()):
        why = ""
        try:
            be = backend_mod.get_backend()
            why = be.why_silent() if hasattr(be, "why_silent") else ""
        except Exception:
            pass
        msg = why or ("The DM didn't answer that one — try again, or press "
                      "Recap and continue if it keeps happening.")
        logger.warning(f"[{sid}] silent turn: {why or 'no reason reported'}")
        # ON SCREEN ONLY, NEVER SPOKEN. The DM's voice belongs to the fiction; having it
        # break character to announce its own failure is worse than the failure -- the table
        # can see perfectly well that nothing happened, and hearing it said out loud stops
        # the game dead. The status line is there for whoever wants a reason.
        await sock.send_text(json.dumps({"type": "status", "msg": "⚠ " + msg}))

    # THE ENGINE IS WAITING ON SOMEBODY. Spoken here, by the server, rather than asked of the
    # model: the ask used to live in the prompt, the model went silent on 30% of turns, and the
    # table was never once told whose turn it was. 433 utterances, 7 end-of-turns, zero asks.
    try:
        import combat as _cb
        if (line := _cb.nudge_line(state)):
            await _say(sock, line, state.campaign)
            await room_mod.broadcast(state.campaign, {"type": "turnwait",
                                                      "who": _cb.waiting_on_pc(state)},
                                     origin=sock)
            state.save()
    except Exception as e:
        logger.warning(f"could not prompt for end of turn ({e})")

    timing = {
        "type": "timing",
        "stt_ms": round(t_stt),
        "llm_first_token_ms": round(t_first_token) if t_first_token else None,
        "first_audio_ms": round(t_first_audio) if t_first_audio else None,   # end-of-speech -> DM voice
    }
    await sock.send_text(json.dumps(timing))
    logger.info(f"[{sid}] timing {timing}")
    transcript.log(state.campaign, state.session, "system", "(turn complete)", route=route,
                   first_audio_ms=timing["first_audio_ms"], stt_ms=timing["stt_ms"])
    return True


if __name__ == "__main__":
    import uvicorn
    scheme = "https" if config.SSL_CERT else "http"
    if not config.SSL_CERT:
        logger.warning("running HTTP -- phone mic will be BLOCKED by iOS. See README (certs/).")
    logger.info(f"serving {scheme}://<this-pc-ip>:{config.PORT}")
    # KEEPALIVE. uvicorn defaults to pinging every 20s and closing the socket if no pong
    # comes back within 20 more, and on a Cloudflare quick tunnel that killed live
    # sessions twice in ninety seconds:
    #
    #   08:40:53 ws-close: code=1011 reason=keepalive ping timeout clean=true
    #   08:41:35 ws-close: code=1011 reason=keepalive ping timeout clean=true
    #
    # The link was not dead. Microphone frames were arriving the whole time -- thirty-two
    # per window, peaking at 0.685 -- so the connection was carrying audio while being
    # closed for failing to answer a control frame. Every reconnect drops the audio in
    # flight and re-arms listening, which is why a phone that was hearing perfectly well
    # never got a single utterance through, and why it looked like a server that had
    # stopped listening.
    #
    # A long timeout rather than no ping at all: a genuinely dead peer should still be
    # collected, it just must not be decided in twenty seconds over a tunnel. The uplink
    # is continuous audio, so real liveness is already proven by the traffic itself.
    uvicorn.run(app, host=config.HOST, port=config.PORT,
                ws_ping_interval=20.0, ws_ping_timeout=120.0,
                ssl_certfile=config.SSL_CERT, ssl_keyfile=config.SSL_KEY)
