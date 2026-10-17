"""Authoritative game state: characters, HP, conditions, and combat/initiative.

This is the source of truth the DM must not hallucinate. The LLM manipulates it only
through tools.py; here we keep the rules honest (damage drops HP, 0 HP downs a PC / kills
a monster, initiative orders turns). Persists to state/<campaign>.json.
"""
import difflib
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import dice

logger = logging.getLogger("state")
STATE_DIR = Path(__file__).parent.parent / "state"
BACKUP_DIR = STATE_DIR / "_backups"
KEEP_BACKUPS = 12


def _peek(path: Path) -> dict | None:
    """Read just enough of the file on disk to judge a pending write against it."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {"rev": int(d.get("rev", 0) or 0), "session": int(d.get("session", 0) or 0),
            "combatants": len(d.get("combatants") or {}),
            "premise": bool(d.get("premise")), "bible": len(d.get("bible") or "")}


BACKUP_EVERY = 300          # seconds; routine saves snapshot at most this often


def _backup(path: Path, why: str = "", throttle: bool = False) -> Path | None:
    """Copy the current file aside. Suspicious writes always snapshot; routine ones are
    throttled, so a campaign keeps a rolling history without a backup per turn."""
    if not path.exists():
        return None
    d = BACKUP_DIR / path.stem
    if throttle and (recent := sorted(d.glob("*.json"))[-1:] if d.exists() else []):
        if time.time() - recent[0].stat().st_mtime < BACKUP_EVERY:
            return None
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    dst = d / f"{stamp}{'-' + why if why else ''}.json"
    shutil.copy2(path, dst)
    for old in sorted(d.glob("*.json"))[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return dst


def _newest_backup(campaign: str) -> dict | None:
    d = BACKUP_DIR / campaign
    for p in sorted(d.glob("*.json"), reverse=True) if d.exists() else []:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


# D&D Beyond stores item names inverted, for sorting: "Clothes, Common", "Crossbow, Light",
# "Pot, Iron", "Longsword, +1". Listed with ", " between items they become unreadable -- the
# DM was handed "equipped: Crossbow, Light, Bolts" and "carrying: Clothes, Common, Pot, Iron"
# and cannot tell a name from a separator, so the party appeared to be carrying an item called
# "Light" and another called "Common". Put the words back the way a person says them.
_INVERTED = re.compile(r"^([A-Za-z][\w'\- ]{2,30}),\s*(\+\d|[A-Za-z][\w'\-]{1,14})$")


def item_name(raw: str) -> str:
    """'Crossbow, Light' -> 'Light Crossbow'. Anything else is returned untouched."""
    m = _INVERTED.match((raw or "").strip())
    return f"{m.group(2)} {m.group(1)}" if m else (raw or "").strip()


def item_list(names) -> str:
    """Join items so the separator can never be mistaken for part of a name."""
    return "; ".join(n for n in names if n)


@dataclass
class Combatant:
    name: str
    max_hp: int = 10
    hp: int = 10
    ac: int = 10
    is_pc: bool = False
    initiative: int | None = None
    init_mod: int = 0
    conditions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)     # e.g. {"str": 16, "dex": 14, ...}
    skills: dict = field(default_factory=dict)    # skill -> "prof" | "expert" (from the sheet)
    level: int = 1
    inventory: list = field(default_factory=list)  # [{name, equipped, qty, ac, armor_type, kind}]
    spells: list = field(default_factory=list)     # [{name, level, school, prepared, source}]
    slots: dict = field(default_factory=dict)      # {"1": {"max": 2, "used": 0}}
    actions: list = field(default_factory=list)    # [{name, kind, damage, equipped}]
    profile: dict = field(default_factory=dict)    # {race, speed, hit_die, cast_stat, subclass}
    traits: list = field(default_factory=list)     # [{name, source, level, text}] racial/class/feat
    pos: list = field(default_factory=list)        # [col, row] on the current battle map
    notes: str = ""
    ddb_id: str = ""                                # so the sheet can be re-read
    pending_level: int = 0                          # announced, not yet done on DDB
    concentrating: str = ""                         # spell they must hold onto
    hit_dice_used: int = 0                          # spent on short rests
    inspiration: bool = False
    resources: dict = field(default_factory=dict)   # player-tracked: slots, potions, inspiration…
    death_saves: dict = field(default_factory=lambda: {"success": 0, "fail": 0})

    @property
    def mods(self) -> str:
        """Ability modifiers, spelled out. The DM is asked 'what do I add?' constantly and
        can only answer if the arithmetic is already in front of it."""
        out = []
        for k in ("str", "dex", "con", "int", "wis", "cha"):
            if (score := self.stats.get(k)) is not None:
                m = (int(score) - 10) // 2
                out.append(f"{k.upper()} {score} ({m:+d})")
        return " ".join(out)

    @property
    def pronouns(self) -> str:
        """How to refer to them. Straight off the D&D Beyond sheet — never inferred from a
        name, which is how the DM came to call a male character 'she'."""
        p = (self.profile or {}).get("pronouns") or ""
        if p:
            return p
        g = ((self.profile or {}).get("gender") or "").strip().lower()
        # DDB's gender box is free text, so it comes back as "Male", "M", "man" and worse.
        # Only these exact forms are read; anything else stays unknown rather than guessed.
        return {"male": "he/him", "man": "he/him", "m": "he/him",
                "female": "she/her", "woman": "she/her", "f": "she/her"}.get(g, "")

    @property
    def vibe(self) -> str:
        """How hurt a monster LOOKS. Players shouldn't be told an enemy's hit points, and the
        surest way to stop that is to never put the number in front of the DM."""
        if self.hp <= 0:
            return "down"
        frac = self.hp / max(self.max_hp, 1)
        if frac >= 0.999:
            return "untouched"
        if frac > 0.75:
            return "barely scratched"
        if frac > 0.5:
            return "hurt"
        if frac > 0.25:
            return "bloodied"
        if frac > 0.1:
            return "badly hurt"
        return "barely standing"

    @property
    def status(self) -> str:
        if self.hp <= 0:
            return "dead" if not self.is_pc else "down"
        frac = self.hp / max(self.max_hp, 1)
        band = "bloodied" if frac <= 0.5 else "healthy"
        c = f" [{', '.join(self.conditions)}]" if self.conditions else ""
        r = "  " + ", ".join(f"{k}:{v}" for k, v in self.resources.items()) if self.resources else ""
        return f"{self.hp}/{self.max_hp} {band}{c}{r}"


class GameState:
    def __init__(self, campaign: str = "default"):
        self.campaign = campaign
        self.premise: str = ""                     # what the campaign is about
        self.roster: list[dict] = []               # players: {player, character, ddb_id, voice_id}
        self.plan: str = ""                         # DM's campaign plan (from planner)
        self.recap: str = ""                        # summary of last session, for continuity
        self.session: int = 0
        self.combatants: dict[str, Combatant] = {}
        self.scene: str = "Session not yet started."
        self.location: str = ""
        self.quests: list[str] = []
        self.bible: str = ""                       # persistent world/lore, grown by the director
        self.ambience: str = ""                    # current soundscape mood
        self.map_wanted: str = ""                  # location the DM asked to be drawn
        self.map_svg: str = ""                     # last drawn battle map (persists for reprint)
        self.map_title: str = ""
        self.map_grid: str = ""                    # the raw tiles, so tokens can be re-placed
        self.map_rooms: list = []                  # [[x, y, name], …] room centres
        self.map_seen: list = []                   # indices of rooms the party has explored
        self.map_dirty: bool = False               # tokens moved / area revealed -> redraw
        self.map_loc: str = ""                     # library key for the current map (raw location)
        self.last_played: float = 0.0              # so reconnecting resumes instead of restarting
        self.resumed: bool = False                 # transient: this connection picked up mid-scene
        self.log: list[str] = []                   # short event log
        self.in_combat: bool = False
        self.round: int = 0
        self.order: list[str] = []                 # names sorted by initiative
        self.pending_init: list[str] = []          # PCs whose own initiative we're waiting on
        # set when a player rolls or takes damage on their own go; combat.brief reads it
        # to know their turn is over without waiting for the model to say so
        self.pc_acted = False
        self.last_subject = ""   # the spell/weapon just named, for "what dice for this"
        self.pending_attack = None  # a confirmed hit still owed a damage roll
        self.asked_end = False     # already asked "is that your turn?" this go
        self.last_target = ""       # who they swung at last, for "I attack it again"
        self.pending_loot: list[str] = []          # found, not yet added on D&D Beyond
        self.sfx_queue: list[str] = []             # one-shots waiting to be pushed
        self.director_event: tuple | None = None   # (event, extra) for the background director
        self.downtime: list[str] = []              # what they did between sessions
        self.lines: list[str] = []                 # content the table has ruled out entirely
        self.veils: list[str] = []                 # allowed, but happens off-screen
        self.turn_idx: int = 0
        self.rev: int = 0                           # bumped on every write
        self._rev_loaded: int = 0                   # what we read; not persisted
        self.rng = random.Random()

    # -- combatants -----------------------------------------------------------
    def add(self, name, max_hp=10, ac=10, is_pc=False, init_mod=0, stats=None, notes="") -> Combatant:
        """Adding someone already here UPDATES them rather than replacing them.

        The DM re-adds a player character to put them in a scene, and a fresh Combatant would
        silently destroy the imported sheet — ability scores, skill proficiencies, inventory,
        level — leaving a character whose own modifiers can no longer be worked out.
        """
        if (existing := self.combatants.get(name.lower())) is not None:
            existing.max_hp = max_hp or existing.max_hp
            existing.hp = min(existing.hp, existing.max_hp) or existing.max_hp
            existing.ac = ac or existing.ac
            existing.is_pc = is_pc or existing.is_pc
            existing.init_mod = init_mod or existing.init_mod
            existing.stats = stats or existing.stats
            existing.notes = notes or existing.notes
            self._log(f"{existing.name} is here ({existing.hp}/{existing.max_hp} HP, AC {existing.ac})")
            return existing
        c = Combatant(name=name, max_hp=max_hp, hp=max_hp, ac=ac, is_pc=is_pc,
                      init_mod=init_mod, stats=stats or {}, notes=notes)
        self.combatants[name.lower()] = c
        self._log(f"{name} enters ({'PC' if is_pc else 'NPC'}, {max_hp} HP, AC {ac})")
        return c

    def rename(self, old: str, new: str) -> str:
        """Give a combatant the name the fiction actually uses.

        Three drowned dig-crew were on the board as "River Wraith A/B/C" while the world bible
        called them Marn, Uld and Coss seventeen, twelve and twelve times over. The DM narrated
        the fiction names and the engine kept answering with the board names, so the two drifted
        apart in the same breath -- and a handover meant for the player came out as "Marn,
        you're up." A creature should have ONE name.
        """
        c = self.get(old)
        if c is None:
            return f"No combatant named {old}."
        new = (new or "").strip()
        if not new:
            return "A name is required."
        if (other := self.get(new)) is not None and other is not c:
            return f"{new} is already on the board."
        self.combatants.pop(c.name.lower(), None)
        was, c.name = c.name, new
        self.combatants[new.lower()] = c
        self.order = [new if n == was else n for n in self.order]
        if getattr(self, "last_target", "") == was:
            self.last_target = new
        if (pa := getattr(self, "pending_attack", None)) and pa.get("target") == was:
            pa["target"] = new
        self.pending_init = [new if n == was else n for n in self.pending_init]
        self._log(f"{was} is called {new}")
        return f"{was} is now {new}."

    def get(self, name) -> Combatant | None:
        return self.combatants.get(name.lower())

    def level_up(self) -> str:
        """Announce a milestone level-up. D&D Beyond does the actual levelling.

        We deliberately do NOT apply the numbers ourselves — DDB already handles multiclass,
        half-caster and pact slot tables, ASIs, feats and subclass features correctly, and two
        systems both editing the same sheet is how they drift apart. So: tell them, then
        verify_levels() re-reads the sheet to confirm they really did it.
        """
        pcs = [c for c in self.combatants.values() if c.is_pc]
        if not pcs:
            return "No player characters to level."
        for c in pcs:
            c.pending_level = (c.level or 1) + 1
        self.save()
        who = ", ".join(f"{c.name} → level {c.pending_level}" for c in pcs)
        return (f"LEVEL UP ({who}). Tell them to level up on D&D Beyond now, then say when "
                "they're done and run [VERIFY] to re-read their sheets. Their numbers do NOT "
                "change until that check passes.")

    def verify_levels(self) -> str:
        """Re-read every pending character from D&D Beyond and confirm they actually levelled."""
        import ddb
        waiting = [c for c in self.combatants.values() if c.is_pc and c.pending_level]
        if not waiting:
            return "Nobody has a level-up pending."
        done, todo, failed = [], [], []
        for c in waiting:
            want, before = c.pending_level, c.level
            r = ddb.resync(c)
            if r.get("error"):
                failed.append(f"{c.name}: {r['error']}")
                continue
            if c.level >= want:
                gained = c.max_hp - r["before"]["max_hp"]
                bits = [f"level {c.level}"]
                if gained:
                    bits.append(f"+{gained} HP (now {c.hp}/{c.max_hp})")
                if c.ac != r["before"]["ac"]:
                    bits.append(f"AC {c.ac}")
                if (new := len(c.spells) - r["before"]["spells"]) > 0:
                    bits.append(f"{new} new spells")
                c.pending_level = 0
                done.append(f"{c.name}: {', '.join(bits)}")
                self._log(f"{c.name} verified at level {c.level}")
            else:
                todo.append(f"{c.name} (still level {before})")
        if done:
            self.save()
        out = []
        if done:
            out.append("VERIFIED — " + "; ".join(done))
        if todo:
            out.append("NOT YET — " + ", ".join(todo) +
                       ". Their sheet still shows the old level; ask them to finish and save on "
                       "D&D Beyond, then [VERIFY] again.")
        if failed:
            out.append("COULDN'T CHECK — " + "; ".join(failed))
        return " ".join(out)

    def damage(self, name, amount, crit=False) -> str:
        c = self.get(name)
        if not c:
            return f"No combatant named {name}."
        amount = max(0, int(amount))
        was_down = c.hp == 0 and c.is_pc
        c.hp = max(0, c.hp - amount)
        msg = f"{c.name} takes {amount} damage -> {c.status}"

        if was_down:
            # a hit on a character who is already down is an automatic death save failure,
            # two if it crit -- the rule everyone forgets, and the one that actually kills PCs
            msg += " " + self._death_fail(c, 2 if crit else 1, "hit while down")
        elif c.hp == 0:
            if c.is_pc:
                c.conditions = list(set(c.conditions + ["unconscious"]))
                c.concentrating = ""
                msg += "  (DOWN -- death saves start on their turn)"
            else:
                msg += "  (SLAIN)"
                self._remove_from_order(c.name)
        elif c.concentrating and amount:
            msg += " " + self._concentration_check(c, amount)
        self._log(msg)
        return msg

    # -- concentration ---------------------------------------------------------
    def concentrate(self, name, spell) -> str:
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        spell = (spell or "").strip()
        prior = c.concentrating
        c.concentrating = spell
        if not spell:
            return f"{c.name} drops concentration."
        # concentration is one spell at a time; starting a second ends the first
        drop = f" ({prior} ends)" if prior and prior.lower() != spell.lower() else ""
        return f"{c.name} is concentrating on {spell}{drop}."

    def _concentration_check(self, c, amount: int) -> str:
        """DC 10 or half the damage, whichever is higher. Forgotten at every real table."""
        dc = max(10, amount // 2)
        con = self._mod(c, "con")
        bonus = con + (self.prof(c) if (c.skills or {}).get("con save") else 0)
        r = dice.roll("1d20", self.rng)
        total = r.total + bonus
        if total >= dc:
            return f"[CONCENTRATION held: {r.total}{bonus:+d} = {total} vs DC {dc}]"
        spell = c.concentrating
        c.concentrating = ""
        return f"[CONCENTRATION BROKEN: {r.total}{bonus:+d} = {total} vs DC {dc} — {spell} ends]"

    # -- death saves -----------------------------------------------------------
    def _death_fail(self, c, n: int, why: str) -> str:
        c.death_saves["fail"] += n
        if c.death_saves["fail"] >= 3:
            return self._dies(c, why)
        return f"[DEATH SAVE {'FAIL x2' if n > 1 else 'FAIL'} ({why}) — {self._saves(c)}]"

    def _dies(self, c, why) -> str:
        c.conditions = [x for x in c.conditions if x != "unconscious"] + ["dead"]
        self._remove_from_order(c.name)
        self._log(f"{c.name} DIES ({why})")
        return f"[{c.name.upper()} IS DEAD — three failed death saves]"

    def _saves(self, c) -> str:
        d = c.death_saves
        return f"{d['success']} success / {d['fail']} fail"

    def death_save(self, name, roll: int | None = None) -> str:
        """One death saving throw. The player can give their own d20; otherwise we roll."""
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        if c.hp > 0:
            return f"{c.name} is conscious — no death save needed."
        if "dead" in c.conditions:
            return f"{c.name} is already dead."
        if "stable" in c.conditions:
            return f"{c.name} is stable — no death saves while stable."
        d = int(roll) if roll else dice.roll("1d20", self.rng).total

        if d == 20:                       # back on their feet with 1 HP, the best moment in D&D
            c.hp = 1
            c.death_saves = {"success": 0, "fail": 0}
            c.conditions = [x for x in c.conditions if x != "unconscious"]
            self._log(f"{c.name} NAT 20 death save — conscious at 1 HP")
            return f"[NATURAL 20 — {c.name} is back up with 1 HP]"
        if d == 1:
            return self._death_fail(c, 2, "natural 1")
        if d >= 10:
            c.death_saves["success"] += 1
            if c.death_saves["success"] >= 3:
                c.death_saves = {"success": 0, "fail": 0}
                c.conditions = list(set(c.conditions + ["stable"]))
                self._log(f"{c.name} stabilises")
                return f"[{d} — {c.name} STABILISES at 0 HP, unconscious but no longer dying]"
            return f"[death save {d} — success, {self._saves(c)}]"
        return self._death_fail(c, 1, f"rolled {d}")

    # -- rests -----------------------------------------------------------------
    def rest(self, kind="long") -> str:
        """A long rest resets the resources we track. Without this the slot counters were
        never cleared, so every number went stale the first time the party slept."""
        pcs = [c for c in self.combatants.values() if c.is_pc and "dead" not in c.conditions]
        if not pcs:
            return "No player characters."
        if kind.lower().startswith("short"):
            lines = []
            for c in pcs:
                avail = (c.level or 1) - c.hit_dice_used
                die = (c.profile or {}).get("hit_die") or 8
                lines.append(f"{c.name} {c.hp}/{c.max_hp}, {avail}d{die} hit dice left")
            self._log("Short rest")
            return ("SHORT REST. " + "; ".join(lines) +
                    ". They may spend hit dice to heal — ask what they roll, then [HITDIE <name> <roll>].")
        lines = []
        for c in pcs:
            c.hp = c.max_hp
            for s in c.slots.values():
                s["used"] = 0
            # a long rest returns half your hit dice, rounded down, minimum one
            c.hit_dice_used = max(0, c.hit_dice_used - max(1, (c.level or 1) // 2))
            c.death_saves = {"success": 0, "fail": 0}
            c.conditions = [x for x in c.conditions
                            if x not in ("unconscious", "stable", "poisoned", "paralyzed")]
            c.concentrating = ""
            lines.append(f"{c.name} {c.hp}/{c.max_hp}")
        self._log("Long rest")
        self.save()
        return ("LONG REST. Full HP and all spell slots back: " + "; ".join(lines) +
                ". Anything they track on D&D Beyond, they should reset there too.")

    def spend_hit_die(self, name, roll: int) -> str:
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        if c.hit_dice_used >= (c.level or 1):
            return f"{c.name} has no hit dice left — only a long rest brings them back."
        c.hit_dice_used += 1
        healed = max(0, int(roll) + self._mod(c, "con"))
        c.hp = min(c.max_hp, c.hp + healed)
        left = (c.level or 1) - c.hit_dice_used
        self._log(f"{c.name} spends a hit die: +{healed} -> {c.hp}/{c.max_hp}")
        return f"{c.name} heals {healed} (roll {roll} + CON) -> {c.hp}/{c.max_hp}. {left} hit dice left."

    # -- inspiration -----------------------------------------------------------
    def inspire(self, name, reason="") -> str:
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        if c.inspiration:
            return f"{c.name} already has inspiration (it doesn't stack)."
        c.inspiration = True
        self._log(f"{c.name} gains inspiration{': ' + reason if reason else ''}")
        return f"[{c.name} GAINS INSPIRATION{' — ' + reason if reason else ''}] Tell them they have it."

    def spend_inspiration(self, name) -> str:
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        if not c.inspiration:
            return f"{c.name} doesn't have inspiration."
        c.inspiration = False
        self._log(f"{c.name} spends inspiration")
        return f"{c.name} spends inspiration — that roll is with advantage."

    # -- loot ------------------------------------------------------------------
    def loot(self, items="", gold=0) -> str:
        """Record a haul. D&D Beyond stays authoritative for what's on the sheet, so this
        logs what was found and leaves the players to add it there — same as levelling."""
        bits = []
        if items:
            bits.append(items.strip())
        if gold:
            bits.append(f"{int(gold)} gp")
        if not bits:
            return "Nothing found."
        entry = ", ".join(bits)
        self.pending_loot.append(entry)
        self._log(f"Loot: {entry}")
        self.save()
        return f"[LOOT: {entry}] Tell them what they found and to add it on D&D Beyond."

    def add_downtime(self, name, activity) -> str:
        """What a character did between sessions. Kept because the payoff is the point: it
        only matters if it comes back later as a contact, a rumour, or a finished item."""
        entry = f"{name}: {activity.strip()}"
        self.downtime.append(entry)
        self.downtime = self.downtime[-12:]
        self._log(f"Downtime — {entry}")
        self.remember(f"Between sessions, {entry}")
        self.save()
        return f"Noted: {entry}. Give them one concrete result of it when the session opens."

    def play_sfx(self, name) -> str:
        n = (name or "").lower().strip()
        self.sfx_queue.append(n)
        return f"(sfx: {n})"

    def _mod(self, c, stat) -> int:
        return ((c.stats or {}).get(stat, 10) - 10) // 2

    @staticmethod
    def prof(c) -> int:
        return 2 + max(0, ((c.level or 1) - 1) // 4)

    def heal(self, name, amount) -> str:
        c = self.get(name)
        if not c:
            return f"No combatant named {name}."
        was_down = c.hp == 0
        c.hp = min(c.max_hp, c.hp + max(0, int(amount)))
        if was_down and c.hp > 0 and "unconscious" in c.conditions:
            c.conditions.remove("unconscious")
            c.death_saves = {"success": 0, "fail": 0}
        self._log(f"{c.name} heals {amount} -> {c.status}")
        return f"{c.name} heals {amount} -> {c.status}"

    def condition(self, name, cond, add=True) -> str:
        c = self.get(name)
        if not c:
            return f"No combatant named {name}."
        cond = cond.lower()
        if add and cond not in c.conditions:
            c.conditions.append(cond)
        elif not add and cond in c.conditions:
            c.conditions.remove(cond)
        self._log(f"{c.name} {'gains' if add else 'loses'} {cond}")
        return f"{c.name}: {c.status}"

    # -- combat ---------------------------------------------------------------
    def start_combat(self, names: list[str] | None = None) -> str:
        """Roll for the monsters; the PLAYERS roll their own initiative.

        This used to roll for everyone, which quietly broke the one rule the whole design
        rests on — players roll their own dice. Combat now waits for their numbers before
        the order exists at all.
        """
        fight = names or list(self.combatants.keys())
        self.pending_init = []
        for key in fight:
            c = self.get(key)
            if not c:
                continue
            if c.is_pc:
                c.initiative = None
                self.pending_init.append(c.name)
            else:
                c.initiative = dice.roll(f"1d20+{c.init_mod}", self.rng).total
        self.in_combat, self.round, self.turn_idx = True, 1, 0
        self.order = []
        if self.pending_init:
            self._log(f"COMBAT begins — waiting on initiative from {', '.join(self.pending_init)}")
            return ("Combat! Roll initiative — waiting on "
                    + ", ".join(self.pending_init)
                    + ". Tell me each number, then I'll set the order.")
        return self._build_order()

    def set_initiative(self, name: str, value: int) -> str:
        """Record a player's own initiative roll (their final total)."""
        c = self.get(name)
        if c is None and self.pending_init:
            # Whoever we are waiting on FIRST. With one player at the table, asking whose
            # number it was is bureaucracy; with four it is a deadlock -- a playtest spent
            # eight turns on "whose roll is that?" while nobody could start the fight. The DM
            # is told to ask one person at a time by name, so an unlabelled number belongs to
            # the person who was asked.
            c = self.get(self.pending_init[0])
        if c is None:
            pcs = [x for x in self.combatants.values() if x.is_pc]
            if len(pcs) == 1:
                c = pcs[0]
        if not c:
            return (f"No combatant named {name}." if name else
                    "Nobody is waiting on an initiative roll.")
        c.initiative = int(value)
        self.pending_init = [n for n in self.pending_init if n.lower() != c.name.lower()]
        self._log(f"{c.name} rolls initiative {c.initiative}")
        if self.pending_init:
            return f"{c.name}: {c.initiative}. Still waiting on {', '.join(self.pending_init)}."
        return self._build_order()

    def _build_order(self) -> str:
        rolled = [c for c in self.combatants.values()
                  if c.initiative is not None and (c.is_pc or c.hp > 0)]
        # ties break toward the higher initiative modifier, then the player
        self.order = [c.name for c in sorted(
            rolled, key=lambda c: (c.initiative, c.init_mod, c.is_pc), reverse=True)]
        self.turn_idx, self.round = 0, 1
        line = ", ".join(f"{nm}({self.get(nm).initiative})" for nm in self.order)
        self._log(f"COMBAT order: {line}")
        return f"Round 1. Order: {line}. Up first: {self.current_turn()}"

    def current_turn(self) -> str | None:
        """Whose go it is, or None. Never raises.

        This indexed self.order directly and threw IndexError whenever turn_idx sat off the
        end -- which a save restored mid-fight, a combatant removed from the order, or a
        hand-edited state can all produce. A crash here takes down the whole turn.
        """
        if not (self.in_combat and self.order):
            return None
        if not 0 <= self.turn_idx < len(self.order):
            self.turn_idx = 0
        return self.order[self.turn_idx]

    def next_turn(self) -> str:
        if not self.in_combat:
            return "Not in combat."
        if not self.order:
            # combat has started but the players haven't given their initiative yet
            return ("Waiting on initiative from " + ", ".join(self.pending_init)
                    if self.pending_init else "No one is in the initiative order yet.")
        self.turn_idx += 1
        if self.turn_idx >= len(self.order):
            self.turn_idx = 0
            self.round += 1
            self._log(f"-- Round {self.round} --")
        nm = self.current_turn()
        c = self.get(nm)
        if c.is_pc and c.hp == 0 and "dead" not in c.conditions and "stable" not in c.conditions:
            # The save is NOT rolled here. combat.py owns death saves now, and rolling in both
            # places burned two of a character's three saves on one turn -- which is half the
            # margin between unconscious and dead, spent invisibly.
            return f"Round {self.round}: {nm} is dying and must make a death saving throw."
        conc = f" (concentrating on {c.concentrating})" if c.concentrating else ""
        # the handover has to be spoken or nobody knows the turn changed; saying so here is
        # far more reliable than hoping the system prompt is still front of mind
        if c.is_pc:
            return (f"Round {self.round}: {nm}'s turn ({c.status}){conc}. Say '{nm}, you're up.' "
                    "out loud and STOP — do not act for them, do not continue.")
        return (f"Round {self.round}: {nm}'s turn ({c.vibe}){conc}. Say whose turn it is, MOVE it "
                "and have it do something with intent, then [NEXTTURN].")

    def end_combat(self) -> str:
        self.in_combat = False
        self._log("Combat ends.")
        return "Combat ended."

    def _remove_from_order(self, name):
        if name in self.order:
            i = self.order.index(name)
            self.order.remove(name)
            if i < self.turn_idx:
                self.turn_idx -= 1
        if self.turn_idx >= len(self.order):
            self.turn_idx = 0

    # -- narrative state ------------------------------------------------------
    def set_scene(self, scene, location=None):
        self.scene = scene
        if location:
            self.location = location
        self._log(f"Scene: {scene[:60]}")

    def set_resource(self, name, key, value, absolute=False) -> str:
        c = self.get(name)
        if not c:
            return f"No character named {name}."
        cur = int(c.resources.get(key, 0))
        c.resources[key] = int(value) if absolute else cur + int(value)
        self._log(f"{c.name} {key} -> {c.resources[key]}")
        return f"{c.name} {key} = {c.resources[key]}"

    def resolve(self, name, pcs_only=False):
        """Snap a spoken name onto a real character.

        Voice transcription mangles fantasy names as a matter of course: one table enrolled
        as Araporno, Gleben, Silvers Raily, Gus and Oktah for characters actually called
        Araporn, Glemben Snirp, Silvers Rayleigh, Gus of The Slippery Rock and Oktah of the
        Wood. Exact lookup matched none of the five, so every seat pointed at nobody.

        Deliberately layered rather than one similarity score, because a bare ratio is both
        too weak and too strong: "gus" scores higher against "glemben snirp" than against
        its own character, while "wolf a" and "wolf b" sit at 0.83 and must never merge.
        """
        want = " ".join((name or "").split()).lower()
        if not want:
            return None
        pool = [c for c in self.combatants.values() if c.is_pc or not pcs_only]
        if (hit := self.combatants.get(want)) and hit in pool:
            return hit
        # one name contains the other: "Gus" <-> "Gus of The Slippery Rock"
        for c in pool:
            n = c.name.lower()
            if n.startswith(want) or want.startswith(n):
                return c
        # same leading word, and only one candidate has it: "Silvers Raily" -> "Rayleigh"
        first = want.split()[0]
        if len(shared := [c for c in pool if c.name.lower().split()[0] == first]) == 1:
            return shared[0]
        # last resort: closest leading word, and only when it is clearly ahead of the rest,
        # so an ambiguous pair resolves to nothing instead of to a coin flip
        scored = sorted(((difflib.SequenceMatcher(None, first, c.name.lower().split()[0]).ratio(), c)
                         for c in pool), key=lambda t: t[0], reverse=True)
        if scored and scored[0][0] >= 0.75 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.15):
            return scored[0][1]
        return None

    def add_player(self, player, character, ddb_id="", voice_id=""):
        # the character name arrives via speech, so resolve it to a real PC before storing it
        if match := self.resolve(character, pcs_only=True):
            character = match.name
        self.roster = [r for r in self.roster if r.get("player") != player]
        self.roster.append({"player": player, "character": character,
                            "ddb_id": ddb_id, "voice_id": voice_id})
        self._log(f"Player {player} joins as {character}")
        return f"{player} is playing {character}."

    def remember(self, note):
        self.bible += ("\n" if self.bible else "") + note.strip()

    def _log(self, s):
        self.log.append(s)
        self.log = self.log[-200:]

    def vocabulary(self) -> str:
        """The campaign's proper nouns, as a sentence to prime speech recognition with.
        Whisper only spells an invented name right if it has seen it in the prompt."""
        names = [r["character"] for r in self.roster]
        names += [c.name for c in self.combatants.values()]
        if self.location:
            names.append(self.location)
        seen, uniq = set(), []
        for n in names:
            if n and n.lower() not in seen:
                seen.add(n.lower()); uniq.append(n)
        if not uniq:
            return ""
        # phrased as prose, not a list -- whisper conditions on it as ordinary text
        return "In this Dungeons & Dragons session we speak of " + ", ".join(uniq[:40]) + "."

    def brief(self) -> str:
        """Just the numbers, for the local quick-answer model. Feeding it the full world bible
        costs prompt-processing time for context it will never use, and every millisecond here
        is the whole point of routing the question locally."""
        lines = []
        for c in self.combatants.values():
            tag = "PC" if c.is_pc else "NPC"
            lines.append(f"{c.name} ({tag}): {c.hp}/{c.max_hp} HP, AC {c.ac}"
                         + (f", {', '.join(c.conditions)}" if c.conditions else ""))
            if c.is_pc and (m := c.mods):
                lines.append(f"  abilities: {m}")
                if c.resources:
                    lines.append("  " + ", ".join(f"{k}: {v}" for k, v in c.resources.items()))
                if worn := [item_name(i["name"]) for i in c.inventory if i.get("equipped")]:
                    lines.append("  equipped: " + item_list(worn))
        if self.in_combat:
            lines.append(f"Combat: round {self.round}, current turn {self.current_turn()}, "
                         f"order {', '.join(self.order)}")
        return "\n".join(lines) or "No characters yet."

    # -- summary for the DM's context ----------------------------------------
    def summary(self) -> str:
        lines = [f"CAMPAIGN: {self.campaign} (session {self.session})",
                 f"LOCATION: {self.location or '?'}", f"SCENE: {self.scene}"]
        if self.premise:
            lines.append(f"PREMISE: {self.premise}")
        if self.roster:
            lines.append("PARTY: " + "; ".join(f"{r['player']} as {r['character']}" for r in self.roster))
        # pronouns come off the D&D Beyond sheet; without them the DM guessed from the name
        if pcs := [c for c in self.combatants.values() if c.is_pc and c.pronouns]:
            lines.append("PRONOUNS (use these, never guess): "
                         + "; ".join(f"{c.name} {c.pronouns}" for c in pcs))
        if self.recap:
            lines.append(f"LAST SESSION: {self.recap}")
        if self.quests:
            lines.append("QUESTS: " + "; ".join(self.quests))
        if self.combatants:
            lines.append("CHARACTERS:")
            for c in self.combatants.values():
                turn = " <-- turn" if self.current_turn() == c.name else ""
                if not c.is_pc:
                    # deliberately no numbers: the DM can only describe what it can see
                    cond = f" [{', '.join(c.conditions)}]" if c.conditions else ""
                    lines.append(f"  {c.name} (NPC): looks {c.vibe}{cond}{turn}")
                    continue
                lines.append(f"  {c.name} (PC): AC {c.ac} {c.status}{turn}")
                if c.is_pc and (m := c.mods):
                    lines.append(f"    {m}")
                if c.is_pc and c.spells:
                    # the DM fumbled a ruling on a spell the character had prepared, because
                    # it had never been told what they can cast
                    prep = [s["name"] for s in c.spells if s.get("prepared")]
                    known = [s["name"] for s in c.spells if not s.get("prepared")]
                    if prep:
                        lines.append("    prepared: " + ", ".join(prep))
                    if known:
                        lines.append("    also knows: " + ", ".join(known))
                    if c.slots:
                        lines.append("    slots: " + ", ".join(
                            f"L{k} {v['max'] - v.get('used', 0)}/{v['max']}"
                            for k, v in sorted(c.slots.items())))
                if c.is_pc and c.inventory:
                    worn = [item_name(i["name"]) for i in c.inventory if i.get("equipped")]
                    carried = [item_name(i["name"]) + (f" x{i['qty']}" if i.get("qty", 1) > 1 else "")
                               for i in c.inventory if not i.get("equipped")]
                    if worn:
                        lines.append("    equipped: " + item_list(worn))
                    if carried:
                        lines.append("    carrying: " + item_list(carried))
        if self.in_combat:
            lines.append(f"COMBAT: round {self.round}, up now: {self.current_turn()}, "
                         f"order {self.order}")
        if self.bible:
            # The bible is append-only and goes out on EVERY turn, so it grows without limit
            # while the player's actual line is ~30 characters. Carry what bears on the scene
            # plus the newest few, and leave the director's own bookkeeping notes out of the
            # fiction entirely. Nothing is deleted -- this trims what is SENT.
            import prompting
            text, _manifest = prompting.relevant_bible(
                self.bible, self.scene, self.location, self.quests)
            lines.append("WORLD NOTES:\n" + (text or self.bible))
        return "\n".join(lines)

    # -- persistence ----------------------------------------------------------
    def _regression(self, disk: dict) -> str | None:
        """Would writing this state destroy a campaign that is plainly richer than it?

        Every real loss here has had the same shape: something built a fresh or stale
        GameState and wrote it over a campaign with sessions of play in it. No legitimate
        save ever drops the whole party or rewinds the session counter, so those are refused
        outright rather than trusted. A deliberate wipe passes force=True and says so.
        """
        if disk["combatants"] and not self.combatants:
            return f"would drop all {disk['combatants']} characters"
        if disk["session"] > self.session:
            return f"would rewind session {disk['session']} to {self.session}"
        if disk["premise"] and not self.premise:
            return "would erase the premise"
        if disk["bible"] and not self.bible:
            return "would erase the world bible"
        return None

    def save(self, touch: bool = True, force: bool = False) -> bool:
        """Write the campaign. Returns False if the write was refused to protect the file.

        Three guarantees, each of which was missing when a campaign got destroyed:
          ATOMIC   — written to a temp file and swapped in, so a crash mid-write cannot
                     truncate a campaign into unreadable JSON.
          REVERSIBLE — the previous contents are copied into state/_backups first whenever
                     the write is in any way suspicious, so nothing is one bad write from gone.
          GUARDED  — a stale or empty state cannot silently overwrite a richer one. save()
                     runs from several places (every turn, the update handler, out-of-band
                     scripts) and last-writer-wins across those was the whole problem.
        """
        # Stamped on every save, and save() already runs after every turn — so however the
        # process ends (closing the app, applying an update, a crash, pulling the plug) the
        # last thing written is also the moment we were last playing, and reconnecting picks
        # the scene back up instead of starting a new session.
        # touch=False writes without claiming we just played — for housekeeping saves.
        if touch:
            self.last_played = time.time()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATE_DIR / f"{self.campaign}.json"
        disk = _peek(path)

        if path.exists() and disk is None:            # unreadable — keep it before replacing
            _backup(path, "corrupt")
        elif disk is not None and force:
            # A deliberate wipe is exactly when a way back matters most. This branch used to
            # fall straight through to the write, so "clear history" — the one caller that
            # forces — was the single operation that left nothing to restore from.
            _backup(path, "forced")
        elif disk is not None:
            if reason := self._regression(disk):
                _backup(path, "refused")
                logger.error(f"REFUSED save of {self.campaign}: {reason}. "
                             f"On-disk copy left intact and backed up.")
                return False
            if disk["rev"] > self._rev_loaded:
                # someone wrote this campaign after we read it; last-writer-wins is fine for
                # play, but the version being replaced must stay recoverable
                _backup(path, "stale")
                logger.warning(f"{self.campaign} changed on disk since load "
                               f"(rev {disk['rev']} > {self._rev_loaded}); backed up.")
            else:
                _backup(path, throttle=True)      # rolling history for ordinary play

        self.rev = max(int(getattr(self, "rev", 0)), disk["rev"] if disk else 0) + 1
        data = {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k not in ("rng", "resumed", "fresh", "director_event")}
        data["combatants"] = {k: asdict(v) for k, v in self.combatants.items()}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)                          # atomic on Windows and POSIX alike
        self._rev_loaded = self.rev
        return True

    @classmethod
    def load(cls, campaign="default") -> "GameState":
        p = STATE_DIR / f"{campaign}.json"
        g = cls(campaign)
        if not p.exists():
            return g
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            # A corrupt file must never become an empty campaign, because the next save would
            # then write that emptiness over the only copy.
            _backup(p, "corrupt")
            if (d := _newest_backup(campaign)) is None:
                raise
            logger.error(f"{campaign} was unreadable ({e}); recovered from backup.")
        for k, v in d.items():
            if k == "combatants":
                g.combatants = {ck: Combatant(**cv) for ck, cv in v.items()}
            elif hasattr(g, k):
                setattr(g, k, v)
        g._rev_loaded = int(getattr(g, "rev", 0))
        return g

    @classmethod
    def backups(cls, campaign: str) -> list:
        d = BACKUP_DIR / campaign
        return sorted(d.glob("*.json"), reverse=True) if d.exists() else []

    @classmethod
    def restore(cls, campaign: str, which: int = 0) -> "GameState":
        """Put a backup back. The current file is itself backed up first."""
        got = cls.backups(campaign)
        if not got or which >= len(got):
            raise FileNotFoundError(f"no backup {which} for {campaign}")
        path = STATE_DIR / f"{campaign}.json"
        _backup(path, "pre-restore")
        shutil.copy2(got[which], path)
        logger.info(f"restored {campaign} from {got[which].name}")
        return cls.load(campaign)
