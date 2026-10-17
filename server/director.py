"""The Director: background specialist sub-agents that make the STORY perform well.

The live DM (llm.py) must be fast, so it can't also do deep world-building mid-sentence.
Instead the Director spawns specialist agents that run ASYNCHRONOUSLY between turns and
write their results into the shared game state (world bible + prepared content). By the
time the DM needs them, the ideas are already there.

Specialists (each a focused Claude persona):
  lorekeeper       -- guards continuity; reconciles new events with established lore
  plot_weaver      -- plans arcs, plants foreshadowing, tracks consequences
  npc_director     -- gives recurring NPCs motives, secrets, and a voice
  encounter_designer-- prepares balanced, thematic encounters ahead of time

Triggers map game events -> which specialists to spawn. All runs are non-blocking:
they enrich FUTURE turns, never stall the current one.
"""
import asyncio
import logging
try:
    from anthropic import AsyncAnthropic
except ImportError:                                # core logic runs without the SDK (mock mode)
    AsyncAnthropic = None
import config

logger = logging.getLogger("director")

SPECIALISTS = {
    "lorekeeper": "You are the Lorekeeper for a live RPG. Guard continuity. Given recent "
        "events and the world bible, output 1-3 terse CANON notes (new facts or reconciliations) "
        "that keep the world consistent. Only durable facts. No prose, just the notes.",
    "plot_weaver": "You are the Plot Weaver. Given the story so far, propose ONE subtle hook, "
        "foreshadow, or consequence the DM could use soon. One or two sentences, concrete and "
        "usable. Do not railroad; offer an option.",
    "npc_director": "You are the NPC Director. For the most relevant recurring NPC in the scene, "
        "give a one-line motive + a secret + a speech tic the DM can voice. Terse.",
    "encounter_designer": "You are the Encounter Designer. Propose ONE thematic encounter fit for "
        "the party's current situation: name, 2-3 foes with rough HP/AC, and a twist. Terse, ready to run.",
    "showrunner": "You are the Showrunner, holding the whole campaign in mind. You are given the "
        "planned ARC and what has actually happened. Say in at most three short lines: where the "
        "party is against the arc, the ONE beat that should land next, and anything the arc set up "
        "that is being forgotten. If they have wandered somewhere better than the plan, say so and "
        "bend the arc to them rather than steering back. Terse, directive, no preamble.",
    "cartographer": "You are the Cartographer. You will be given a location and the number of "
        "distinct areas on a battle map of it. Name each area in AT MOST TWO WORDS, evocative and "
        "specific to this place. Then add ONE line: a tactical feature a clever player could turn "
        "to their advantage. Output the names one per line, then the feature prefixed 'FEATURE:'.",
}

# game event -> specialists to spawn
TRIGGERS = {
    "session_start": ["showrunner", "lorekeeper", "plot_weaver"],
    "scene_change":  ["showrunner", "npc_director", "cartographer"],
    "downtime":      ["plot_weaver", "encounter_designer"],
    "combat_end":    ["lorekeeper", "showrunner"],
    "explore":       ["encounter_designer", "npc_director"],
}


# Specialists whose output is about WHERE THE STORY GOES. Only these are worth gating on
# drift -- a map still has to be drawn and an NPC still has to have a voice when the party
# walks into a room, whether or not that room was in the plan.
_STORY_ROLES = {"showrunner", "plot_weaver", "lorekeeper"}

# Debounce lives at MODULE level, keyed by campaign and role, because a Director does not
# outlive a WebSocket connection. session_start fires on every connect, so a phone that
# reconnected -- a locked screen, a tunnel blink, a page refresh -- built a fresh Director
# with an empty debounce table and paid for three more Opus calls. The log shows every
# specialist running exactly twice, and "23 arc beats primed" logged twice a second apart.
_LAST_RUN: dict[tuple[str, str], float] = {}
# Spend, per CAMPAIGN, not per Director. A Director does not outlive a websocket and phones
# reconnect constantly -- a locked screen, a tunnel blink, a reload. One real session opened
# 130 connections, each with a fresh budget of 40, and burned 117 Opus calls against a cap
# that was supposed to stop at 40. The debounce was already module-level for exactly this
# reason; the budget was not, and it was the half that cost money.
_SPENT: dict[str, int] = {}


class Director:
    def __init__(self, state):
        self.state = state
        self.client = (AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
                       if AsyncAnthropic and config.ANTHROPIC_API_KEY else None)
        self.model = getattr(config, "DIRECTOR_MODEL", config.DM_MODEL)
        self.map_prep: dict = {}                   # location -> room names, ready early
        # self._calls is gone: see _SPENT above. A per-instance counter reset on every
        self._warned = False
        self.skipped = 0        # story passes drift said were unnecessary
        self._pending: set[asyncio.Task] = set()
        self.prepared: list[str] = []              # ready-to-use ideas the DM can pull from

        # Embed the arc now, in the background, so the first drift check of the session is a
        # 10ms comparison rather than a 7s model load in front of a waiting player.
        try:
            import drift
            asyncio.get_running_loop().run_in_executor(None, drift.prime, state)
        except RuntimeError:
            pass                                   # no loop yet (tests); primed on first use

    def on_event(self, event: str, extra: str = ""):
        """Fire-and-forget: spawn the specialists for this event in the background.

        Budgeted and debounced. One session fired 119 Opus calls -- three specialists on
        nearly every event -- and the log shows showrunner running TWICE inside the same
        second on the same event, producing two near-identical notes. Duplicate work is
        invisible because it happens off the critical path, so it never got noticed while
        being paid for on every turn.
        """
        import time
        now = time.monotonic()
        roles = TRIGGERS.get(event, [])
        if not roles:
            return
        # Does the STORY need re-aiming, or is the party walking the plan we already wrote?
        # Firing on the event alone spent four showrunner passes in session 9 re-deciding an
        # arc the party never left. drift.py answers that locally in about 10ms; when it says
        # they are on the arc, the plan is already in the prompt and another pass at it would
        # only restate what the DM can read.
        if any(r in _STORY_ROLES for r in roles):
            try:
                import drift
                go, why = drift.needs_director(self.state, event, extra)
            except Exception as e:                 # never let the cheap check break the layer
                go, why = True, f"drift check failed ({e})"
            if not go:
                logger.info(f"director: {event} skipped — {why}")
                self.skipped += 1
                roles = [r for r in roles if r not in _STORY_ROLES]
                if not roles:
                    return
            else:
                logger.debug(f"director: {event} runs — {why}")
        for role in roles:
            camp = getattr(self.state, "campaign", "default")
            if _SPENT.get(camp, 0) >= config.DIRECTOR_BUDGET:
                if not self._warned:
                    self._warned = True
                    logger.info(f"director budget spent ({config.DIRECTOR_BUDGET} calls); "
                                f"the DM keeps everything it already has")
                return
            # Debounce on the ROLE alone, not on (role, event). A specialist does the same job
            # whichever event woke it, so showrunner on session_start followed by showrunner on
            # scene_change a second later is the same note written twice -- which is precisely
            # what the log showed. Keying on the pair let that through.
            key = (getattr(self.state, "campaign", ""), role)
            if now - _LAST_RUN.get(key, -1e9) < config.DIRECTOR_DEBOUNCE_S:
                logger.info(f"director: {role} debounced (ran recently, same campaign)")
                continue
            _LAST_RUN[key] = now
            _SPENT[camp] = _SPENT.get(camp, 0) + 1
            task = asyncio.create_task(self._run(role, event, extra))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def _run(self, role: str, event: str, extra: str):
        ctx = f"EVENT: {event}\n{extra}\n\n--- game state ---\n{self.state.summary()}\n\n"
        # "on track" is measured against the arc, and state.summary() does not carry it
        if plan := getattr(self.state, "plan", ""):
            ctx += f"--- the planned ARC ---\n{plan}\n\n"
        ctx += "--- recent events ---\n" + "\n".join(self.state.log[-12:])
        try:
            out = await self._ask(role, ctx)
        except Exception as e:
            logger.warning(f"{role} failed: {e}")
            return
        if not out:
            return
        if role == "cartographer":
            # Geometry stays in Python (instant, deterministic). This is only the DRESSING —
            # names and one tactical feature — worked out before the fight rather than during.
            self.map_prep[(extra or "").strip().lower()] = out.strip()
            logger.info(f"director/cartographer: prepped {extra!r}")
            return
        tag = f"[{role}] {out.strip()}"
        if role in ("lorekeeper",):
            self.state.remember(tag)               # canon -> permanent bible
        else:
            self.prepared.append(tag)              # ideas the DM may pull in
            self.prepared = self.prepared[-12:]
        logger.info(f"director/{role}: {out.strip()[:80]}")

    async def _ask(self, role: str, ctx: str) -> str:
        if self.client is not None:                # API path (fastest, if a key is set)
            msg = await self.client.messages.create(
                model=self.model, max_tokens=200,
                system=SPECIALISTS[role],
                messages=[{"role": "user", "content": ctx}])
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        # "hybrid" routes quick lookups to the local model but still drives the STORY
        # through Claude Code. Testing only for "claude_code" here meant the entire
        # director layer silently returned nothing the moment the backend went hybrid.
        if config.DM_BACKEND in ("claude_code", "hybrid"):
            return await self._ask_claude_code(role, ctx)
        return ""                                  # no backend configured -> add nothing

    async def _ask_claude_code(self, role: str, ctx: str) -> str:
        """One-shot Claude Code call on the Pro subscription. Runs off the real-time path
        (fired between turns), so its cold-start latency never stalls the live DM."""
        import telemetry
        with telemetry.Turn("claude_code", config.DIRECTOR_CLAUDE_MODEL,
                            getattr(self.state, "campaign", ""), route=f"director/{role}") as tel:
            tel.prompt(system=SPECIALISTS[role], message=ctx)
            out = await self._spawn(role, ctx)
            tel.first_token(); tel.chunk(out)
            return out

    async def _spawn(self, role: str, ctx: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            config.CLAUDE_BIN, "-p", "--model", config.DIRECTOR_CLAUDE_MODEL,
            "--output-format", "text", "--append-system-prompt", SPECIALISTS[role],
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(ctx.encode()), timeout=90)
        return out.decode(errors="ignore").strip()

    def pull_prepared(self) -> str:
        """Give the DM the queued ideas, plus the scene written ahead for where they are.

        Two different things share this channel and they are not interchangeable. The queued
        ideas are REACTIVE -- a specialist noticed something a moment ago -- and are consumed
        once. The prepared scene is STANDING material for the beat the party is walking into,
        written by Opus before the session, and it has to survive being read on every turn
        they spend in that scene, so it is deliberately not cleared.
        """
        parts = []
        if self.prepared:
            parts.append("PREPARED IDEAS (use if they fit):\n" + "\n".join(self.prepared))
            self.prepared = []
        try:
            import prepared as prepared_mod
            if scene := prepared_mod.brief(self.state):
                parts.append(scene)
        except Exception as e:                 # prep must never break a live turn
            logger.debug(f"prepared scene unavailable: {e}")
        return ("\n\n").join(parts)

    async def drain(self, timeout=8.0):
        """Await outstanding specialists (e.g., at session boundaries or in tests)."""
        if self._pending:
            await asyncio.wait(self._pending, timeout=timeout)
