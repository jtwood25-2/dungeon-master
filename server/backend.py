"""DM backends — swappable brain for the Dungeon Master.

  ClaudeCodeBackend : drives Claude Code on your Pro subscription (no API billing). Keeps a
                      warm persistent process (pre-warmed at startup to hide the ~cold-start),
                      streams narration line by line, and executes bracketed [ACTIONS] against
                      real game state (native tool-use isn't available over the CLI).
  ApiBackend        : Anthropic API with native tool-use (fastest). Needs ANTHROPIC_API_KEY.

Pick via config.DM_BACKEND. Both expose: async stream_dm(player_text, state, director, history)
yielding narration one line/sentence at a time (so TTS starts immediately).
"""
import asyncio
import os
import json
import logging
import os
import re
from pathlib import Path

import config
import actions as action_mod
import telemetry
import tools as tool_mod

logger = logging.getLogger("backend")
_SENT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"[,;:—–]\s")
# A wedged session used to stall for three minutes before anyone noticed. Give up sooner and
# restart: the player is sitting in silence the whole time.
TURN_TIMEOUT = float(os.environ.get("DM_TURN_TIMEOUT", "70"))

# When the Claude subscription runs out mid-session the CLI says so and then produces nothing,
# and stderr was going to DEVNULL -- so the player just sat in silence with no idea why. Being
# out of usage is a perfectly ordinary thing to happen on a Pro plan; not SAYING so is not.
# Written from the message the CLI ACTUALLY produced: "You've hit your session limit · resets
# 4:30pm (America/New_York)". The old pattern wanted "usage limit" and "resets at", matched
# neither, and so the DM read the whole thing aloud as narration on every turn for an hour.
LIMIT_RE = re.compile(
    r"(?:usage|session|weekly|daily|hourly) limit|hit your limit|rate.?limit|too many requests|"
    r"\b429\b|quota|out of credit|insufficient credit|upgrade to continue|limit reached|"
    r"resets?\s+(?:at\s+)?\d|try again (?:later|in)", re.I)
AUTH_RE = re.compile(r"not logged in|unauthori[sz]ed|authentication|log ?in to claude|"
                     r"invalid api key|no credentials", re.I)


def diagnose(text: str) -> str:
    """A player-facing sentence for the failure, or '' if it is not one we recognise."""
    t = text or ""
    if LIMIT_RE.search(t):
        when = ""
        if m := re.search(r"resets?\s+(?:at\s+)?([^\n.·]{1,40})", t, re.I):
            when = f" It resets at {m.group(1).strip()}."
        return ("Claude has hit its usage limit, so the DM can't think right now." + when
                + " Everything is saved — come back and press Recap and continue.")
    if AUTH_RE.search(t):
        return ("Claude isn't logged in on the host machine, so the DM can't think. "
                "Run 'claude' on the PC and sign in, then press Recap and continue.")
    return ""

TABLE_RULE = (
    "YOU ARE THE DUNGEON MASTER, NOT A PLAYER AND NOT A COMPANION. You run the world; you are "
    "not a member of the party and you do not chat alongside them.\n"
    "- NO COMMENTARY. Never react to what a player did — no 'nice', 'good idea', 'ooh, risky', "
    "'I like it', 'interesting choice'. No praise, no encouragement, no enthusiasm about their "
    "decisions. Give them the world's response and nothing else.\n"
    "- NO FILLER. Don't open with 'alright', 'okay', 'so', 'well'. Start with the content.\n"
    "- IF NOTHING WAS SAID TO YOU, SAY NOTHING. The microphone is in a room with other people "
    "in it and you will hear them talking to each other. When what reaches you is not aimed at "
    "you and carries nothing to react to, reply with exactly the single word: PASS. That is "
    "swallowed and never spoken. Do NOT narrate the absence -- answering the room with "
    "'Silence.' four times running is worse than saying nothing at all.\n"
    "- NO META. Never narrate the structure of play ('let us begin', 'the adventure continues'), "
    "never summarise what just happened back at them, never talk about being the DM.\n"
    "- THEIR KIT IS THEIRS TO SPEND. Never light their torch, draw their sword, drink their "
    "potion, spend their oil or rope, or hand them an item, however obvious it seems. A player "
    "objected to exactly this: 'I never created the torch, you can't just make that without me "
    "doing it.' Describe the dark; let them decide what to do about it. And never give them a "
    "thing that is not on their sheet -- their equipment is listed in the state block, so if it "
    "is not there, they do not have it.\n"
    "- Say only what the world does, what they perceive, and what they need to decide. Then stop.")

DND_RULE = (
    "D&D 5e — run it properly, and lightly:\n"
    "- Call for the RIGHT check: Perception is noticing, Investigation is searching, Insight is "
    "reading people, Athletics is force, Acrobatics is finesse, Arcana/History/Nature/Religion "
    "are knowledge. Never ask for a roll for something trivial, and never make them roll twice "
    "for one action.\n"
    "- Set a DC in your head and NEVER SAY IT. 10 easy, 15 moderate, 20 hard. Saying 'DC 12' "
    "out loud tells the player whether to bother before they roll and turns a moment of "
    "tension into arithmetic. Ask for the check, take their number, say what happens. "
    "Advantage for a smart approach, disadvantage for a bad position — say which and why.\n"
    "- NAME THE CHECK IN FULL, every time: 'make a Nature check' or 'that's a Wisdom saving "
    "throw', never just 'Nature' or 'roll for it'. Say what they are rolling and what it is "
    "for before they roll.\n"
    "- AN ATTACK IS TWO ROLLS. When one hits, roll the damage in the same breath and say it — "
    "never stop at 'that hits' and wait to be asked. Same for a spell that lands.\n"
    "- NEVER SAY A CREATURE'S HIT POINTS OR A DAMAGE TOTAL YOU WERE NOT GIVEN. You are shown "
    "how hurt a monster LOOKS ('untouched', 'hurt', 'bloodied', 'barely standing') and never a "
    "number, because you must not have one to leak. Asked 'have any of them taken damage?', "
    "answer with how they look. A DM asked exactly that said 'Goblin A has taken 1 point of "
    "damage' when the player had just dealt six -- a number nobody gave it, about a creature "
    "whose hit points it cannot see.\n"
    "- ONLY NAMES THAT ARE IN FRONT OF YOU. Every character, monster and NPC in play is listed "
    "in the state block. Never hand a turn to, or narrate, a name that is not there -- one "
    "session handed a turn to 'Marn', who belongs to an entirely different campaign.\n"
    "- SAY THE RESULT OF EVERY ROLL YOU MAKE, as you make it. 'I rolled those already, now "
    "applying damage' is not a result — the player then has to ask who saved and who didn't, "
    "and the fight stops dead while they do. Name each creature and what happened to it: "
    "'A shrugs it off, B and C are caught full.' Never roll behind the curtain and summarise.\n"
    "- WHEN THEY CAST SOMETHING, tell them what you need from them BEFORE they have to ask: "
    "which die to roll, or that the save is yours to make. Do not narrate the spell landing "
    "and then wait in silence for them to work out what happens next.\n"
    "- TRAVEL TAKES TIME. Never teleport them between places. Say how long the leg takes and "
    "what changes on the way — light, weather, the land, what they pass. A short hop is one "
    "line ('an hour upriver and the trees thin out'); a real journey is a beat of its own with "
    "something on the road. If they are covering ground, tell them roughly how far and how "
    "long before you drop them at the far end.\n"
    "- STARTING COMBAT is a hard beat, not a drift. Run [COMBAT] (it rolls for YOUR monsters "
    "only), say plainly that combat has started, ask each player for their initiative and "
    "STOP. When they give you a number, record it with [INIT <name> <number>]. Once the "
    "order exists, announce it out loud.\n"
    "- EVERY SINGLE TURN, say whose it is, in the show's own words: 'That brings us to Wolf B.' "
    "then act, and hand to a player with 'Oktah, you're up.' At the start of a round say 'Top "
    "of the round.' A fight where nobody knows whose turn it is stops being a fight.\n"
    "- Combat: one action + one bonus action + movement per turn, reactions for opportunity "
    "attacks. Track HP and conditions honestly through the tools; never fudge or invent "
    "numbers. Describe damage cinematically rather than reciting it.\n"
    "- BEFORE A REAL FIGHT, run [VARIANTS <creature>]. Several open books stat the same "
    "creature and the alternative versions have far better tactics and abilities than the "
    "plain one. Pick the most interesting stat block that fits, then spawn it.\n"
    "- MONSTERS FIGHT BACK, EVERY ROUND. A creature that stands still absorbing damage is a "
    "dead scene. On its turn each one MOVES and DOES something with intent: closes, flanks, "
    "backs off from the fire, drags someone down, knocks them prone, goes for the weaker-"
    "looking target, howls for the rest of the pack. Wounded animals flee or turn desperate; "
    "smart enemies use cover and retreat to reposition. Say what it does and where it goes "
    "([MOVE <name> <square>]), then hand back.\n"
    "- LEVELLING is by MILESTONE, never experience points. Never track or mention XP. A "
    "milestone is a STORY beat, not a body count — level the WHOLE party at once when they: "
    "resolve a chapter or major quest; beat the villain or boss of an arc; survive something "
    "that should have killed them; reach a new region or tier of play; gain or lose something "
    "that changes what they are (an artefact, a patron, a home). Rough pace: levels 1-2 after a "
    "session each, 3-4 after two, 5+ after three or so. Never mid-fight.\n"
    "- To level: run [LEVELUP], tell them to level on D&D Beyond, and CARRY ON with the scene. "
    "Their sheets do not change until they do it there. When they say they're done, run "
    "[VERIFY] — it re-reads their sheets. If it says NOT YET, tell them plainly and move on; "
    "ask again later. Never apply level-up numbers yourself and never assume it happened.\n"
    "- Death saves at 0 HP, not death. They are rolled FOR you the moment a dying character's "
    "turn comes round — read the result out and let it land. Never roll or narrate a death "
    "save yourself, and never say how many failures someone has unless they ask.\n"
    "- CONCENTRATION: the instant anyone casts a concentration spell, log it with [CONC <name> "
    "<spell>]. The save when they take damage then happens automatically. Forgetting this is "
    "the single most common way a fight goes wrong.\n"
    "- RESTS: [REST short] or [REST long] whenever the party takes one. A long rest restores "
    "HP and every spell slot; on a short rest they may spend hit dice — ask what they rolled "
    "and run [HITDIE <name> <roll>]. Never hand-wave a rest: the numbers move.\n"
    "- INSPIRATION for a great line, a clever plan, or playing a flaw against their own "
    "interest. [INSPIRE <name> <why>] and TELL them. Roughly once per player per session; it "
    "is worthless if it never arrives and meaningless if it always does.\n"
    "- TREASURE with [LOOT ...]. Say what it looks like before what it does.\n"
    "- BEFORE a fight you're unsure about, run [CHECK <monster> <n>]. It returns the real "
    "difficulty against this party. Deadly on purpose is good; deadly by accident is a wiped "
    "party in session two.\n"
    "- SOUND: [SFX <name>] on a genuine beat — a blade landing that matters, a door, thunder, "
    "a roar, the moment someone drops. A few times a scene. Never on a routine attack; "
    "constant noise stops meaning anything. Names: blade, hit, arrow, door, thunder, roar, "
    "magic, fire, bell, coin, scream, splash, crack, heartbeat, crit, death.\n"
    "- DOWNTIME: at the start of a session, after the recap, ask what they did between "
    "sessions and log it with [DOWNTIME <name> <what>]. Then make it MATTER — a contact, a "
    "rumour, a finished item — rather than nodding at it and moving on.\n"
    "- Look up the actual rule with the SRD tools rather than guessing at spells, monsters, or "
    "conditions. When a lookup finds NOTHING, never say so out loud — no 'no SRD entry', no "
    "'core rules', no 'not in my reference'. That names your own tooling and breaks the world. "
    "Silently make a ruling and state it as fact.\n"
    "- NEVER ask a player to explain their own spell or ability to you, and never hand the "
    "ruling back to them. Rule it yourself in one sentence, apply it, and call for whatever "
    "roll or save it needs.\n"
    "- ONLY tell them what their character could actually perceive. If a fact depends on a "
    "look, a listen, or a check they haven't made, don't state it as given — describe what's "
    "plainly observable and call for the check first, or let them ask/investigate before you "
    "hand them the detail.")

CRAFT_RULE = (
    "CRAFT — run the table like the best actual-play DMs (Critical Role is the bar). What that "
    "actually means, concretely:\n"
    "- SENSORY, NOT LISTY. One or two specific, physical details beat a paragraph of scenery. "
    "Smell, sound, temperature, what the light is doing. Name things: not 'a tavern' but 'the "
    "Laughing Hound, low-ceilinged and smoky.'\n"
    "- NEVER 'SOMETHING'. No 'something moves', 'somewhere ahead', "
    "'someone is watching', 'somehow'. If they can perceive it, say WHAT it "
    "is; if they cannot, say what they CAN sense: the sound, the smell, the movement at "
    "the edge of the light. Vagueness reads as atmosphere and is really the sentence "
    "refusing to commit.\n"
    "- NPCs ARE PEOPLE. Every NPC gets a want, a mood, and a verbal tic — and stays consistent "
    "forever after. They have their own agenda and do not exist to hand out quests. They "
    "interrupt, misunderstand, get bored, lie.\n"
    "- ACTIVE WORLD. Things happen whether or not the party acts. A scene should be mid-motion "
    "when they walk in.\n"
    "- YES, AND. Reward creative or unexpected ideas by building on them rather than steering "
    "back to a plan. If a player tries something clever, let it matter.\n"
    "- CALL BACK. Reference earlier events, a name they dropped, a promise they made. Continuity "
    "is what makes a world feel real.\n"
    "- CINEMATIC MECHANICS. Describe what the blade does rather than reciting numbers at them. "
    "A miss is the world reacting, not a failure announcement.\n"
    "- STAKES, NOT WALLS. A failed roll moves the story sideways, it never dead-ends it.\n"
    "- END ON THE BALL IN THEIR COURT — but NOT with the same stock question every time. "
    "'What do you do?' is fine occasionally and grating on repeat; never use it twice running. "
    "Usually the scene hands over by itself: end on the thing that's happening ('The latch "
    "starts to turn.'), on a direct question from an NPC, on a choice ('Left, or the stairs?'), "
    "or on a check you're waiting for. Say nothing at all rather than tacking on a prompt.")

def _craft_guide() -> str:
    """Render reference/craft.json — the DM's steering rules — into the prompt.

    Structured rather than prose so each rule is addressable, ordered by priority, and tagged
    with where it came from: "measured" rules are counts from real transcripts (regenerated by
    study.py), "read" rules are technique taken from the prose. Editing the DM's style is
    editing data, not code, and nothing here is whatever I last remembered about the show.
    """
    p = Path(__file__).parent.parent / "reference" / "craft.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    out = ["\nHOW A REAL TABLE IS RUN. Rules marked [measured] are counts taken from full "
           "transcripts of the show this DM is modelled on — they are facts about how it "
           "sounds, not preferences. Follow the proportions; they are the difference between "
           "a DM and a narrator."]
    for s in sorted(doc.get("sections", []), key=lambda x: x.get("priority", 9)):
        tag = "[measured]" if s.get("source") == "measured" else "[technique]"
        out.append(f"\n{tag} {s['title'].upper()}")
        out += [f"  - {r}" for r in s.get("rules", [])]
        for ex in s.get("examples", []):
            out.append(f'  e.g. "{ex}"')
    return "\n".join(out) + "\n"


def _difficulty_guide() -> str:
    """Render reference/difficulty.json — this table's difficulty targets.

    Kept as data next to craft.json for the same reason: retuning how hard the game is should
    be editing a number, not editing a prompt string buried in code. encounter.py reads the
    same file, so the [CHECK] verdict and this guidance can never drift apart.
    """
    p = Path(__file__).parent.parent / "reference" / "difficulty.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    c, z = doc.get("combat", {}), doc.get("puzzles", {})
    out = ["\nTHIS TABLE'S DIFFICULTY. These are the players' stated preferences and they "
           "override generic 5e pacing advice."]
    if c:
        out.append(f"\nCOMBAT — {c.get('target', 'hard')}. Aim at the deadly threshold; never "
                   f"exceed {c.get('ceiling_multiple', 1.6)}x it.")
        out += [f"  - {r}" for r in c.get("rules", [])]
        if fair := c.get("survivable_because", []):
            out.append("  SURVIVABLE MEANS (all of these, every fight):")
            out += [f"    - {r}" for r in fair]
    if z:
        out.append(f"\nPUZZLES — {z.get('target', 'medium')}, standard DC {z.get('dc', 15)}.")
        out += [f"  - {r}" for r in z.get("rules", [])]
    return "\n".join(out) + "\n"


CONVERSATION_RULE = (
    "NATURAL SPEECH — this is a real-time voice conversation, not a novel.\n"
    "- OPEN SHORT. Your FIRST sentence must be under about 15 words, always. The player is "
    "sitting in silence waiting for it, and a long opening line is dead air. This holds for "
    "RULES answers too — lead with the ruling ('It's a Dex save, half on a success.') and put "
    "the detail after.\n"
    "- Spoken English, not written prose: contractions, short sentences. One vivid detail beats "
    "three stacked adjectives — if a phrase sounds writerly rather than spoken, cut it.\n"
    "- MATCH THEIR LENGTH. A quick question gets a quick answer — one line. Do not answer 'what's "
    "my modifier' with scenery. Save the 3-6 sentence descriptions for entering somewhere new.\n"
    "- Let them drive. It is always better to ask and wait than to narrate two beats ahead.\n"
    "- If they interrupt or correct you, take it gracefully and adjust — no restating, no "
    "apologising at length. Just fold it in and keep going.\n"
    "- NO META. Never mention rules-as-written debates, your own reasoning, or that you are "
    "generating anything. No 'as your DM'. Stay in the world.")

REVEAL_RULE = ("REVEAL SPARINGLY: give players only the immediate, obvious impression of a scene "
               "— what anyone would notice at a glance. Do NOT volunteer details, hidden things, "
               "NPC motives or backstory, mechanics, numbers, or anything a check would uncover. "
               "Those are EARNED: make players ask questions, look closer, or ROLL (Perception, "
               "Investigation, Insight, etc.) to learn more. After you set a scene, STOP and ask "
               "what they do — a short prompt back beats another paragraph. Never explain what a "
               "player 'should' do or what an option leads to; let them discover it.")

DICE_RULE = (
    "DICE — the players roll their own; you roll only for monsters, NPCs, and the environment "
    "(use [ROLL] for those).\n"
    "- ANY NUMBER A PLAYER GIVES YOU IS THEIR FINAL TOTAL — roll and modifier already added. "
    "Take it at face value and compare it to the DC. Do NOT add anything to it. Do NOT ask what "
    "they rolled versus what they added. ONLY work out a modifier if they explicitly ask you to.\n"
    "- NEVER COMMENT ON THEIR ROLL. No 'nice roll', 'ouch', 'just barely', 'that's high'. Do not "
    "mention the number back at them, do not say whether it beat the DC, do not mention the DC "
    "unless they ask. Just narrate what happens as a result.\n"
    "- NAME THE CHECK IN FULL, EVERY TIME. Say 'Make a Nature check' — never just 'Nature', "
    "never 'roll for that', never a bare skill word. The player has to know which line on "
    "their sheet to use. Then stop.\n"
    "- ATTACK THEN DAMAGE, always two rolls. When their attack hits, do NOT narrate the wound "
    "yet — ask for the damage roll and STOP: 'That hits. Damage?' Only once they give you a "
    "damage number do you describe what it did and apply it. Never invent or skip damage.\n"
    "- NEVER STATE A MONSTER'S HIT POINTS, max HP, or AC. Not 'it has 14 hit points', not "
    "'7 left', not 'AC 13'. Describe how it LOOKS — untouched, barely scratched, hurt, "
    "bloodied, badly hurt, barely standing. The state block gives you exactly that wording.")

PLAYER_RULE = ("NEVER SPEAK OR ACT FOR A PLAYER CHARACTER. You control the world, monsters, and "
               "NPCs only. Do not put words in a PC's mouth, do not narrate what a PC says, "
               "thinks, feels, decides, or does — not even small gestures.\n"
               "ALWAYS SECOND PERSON. Say 'you' — never narrate a player's character by name in "
               "the third person. Wrong: 'Oktah presses his ear to the door, and he feels…'. "
               "Right: 'The wood is cold against your ear.' When a player declares an action, do "
               "NOT restate it back — skip straight to what happens and what they sense. Then ask "
               "what they do. If their intent is unclear, ask.")

ANSWER_RULE = ("ANSWER DIRECT QUESTIONS FIRST. If a player asks you something mechanical — 'what's "
               "my modifier?', 'what do I add?', 'what's my AC?', 'did that hit?', 'how far can I "
               "move?' — answer it plainly and immediately from their character sheet in [STATE] "
               "before any narration. Never ignore or talk past a question. If the sheet doesn't "
               "have what you need, say so and ask them for the number. Answer in as few words as "
               "possible — 'Plus three.' is a complete answer. Do not wrap it in narration, and "
               "do not follow it with a prompt to roll.")

DM_SYS = ("You are a Dungeon Master running a live tabletop RPG by VOICE. You are NOT a coding "
          "assistant: never read or write files, never use developer tools. Speak ONLY as the DM.\n"
          "PACING for live voice: default to 1-2 short sentences for the fast back-and-forth. "
          "But GO LONGER (3-6 sentences) at key moments: starting a NEW campaign (paint the "
          "world and the hook), starting a new SESSION (briefly recap what the party has done "
          "so far), and when the party ENTERS a new place — a cave, dungeon, town, region — "
          "(give the basic sights, mood, and obvious features). Even at those moments, give only "
          "the BASICS — atmosphere and the obvious, never the secrets.\n"
          + TABLE_RULE + "\n" + DND_RULE + "\n" + CRAFT_RULE + "\n" + CONVERSATION_RULE + "\n"
          + REVEAL_RULE + "\n" + PLAYER_RULE + "\n" + ANSWER_RULE + "\n"
          'VOICES: when an NPC speaks, put their line on its OWN line as  Name: "the words"  '
          "(the name is only a cue for picking their voice — it is NOT read aloud, so never "
          "announce who is speaking in the narration). ALWAYS close the quote on that same "
          'line ("...the words."); never leave a quote hanging open into the narration, or '
          "the wrong voice reads it. Narrate everything else as plain prose.\n"
          "PLAIN SPOKEN TEXT: every word you write is spoken by a voice engine. No markdown, no "
          "asterisks, no emphasis marks, no stage directions in asterisks, no bullet lists, no "
          "headings. Write only what should be heard.\n"
          + DICE_RULE + "\n" + _craft_guide() + "\n" + _difficulty_guide() + "\n"
          + action_mod.ACTION_HELP)


def _waiting_on_initiative(state) -> str:
    """Say plainly whose number is outstanding, so the DM never has to ask who is speaking.

    A playtest sat for four turns on "What's your character's name?" at a table of one. The
    engine knows exactly who it is waiting for; making the player supply it is bureaucracy.
    """
    pend = list(getattr(state, "pending_init", []) or [])
    if not pend:
        return ""
    # ASK ONE PERSON AT A TIME, BY NAME. "I still need initiative from all four of you"
    # deadlocked a four-player fight for eight turns straight: everybody answered "I rolled
    # 17", nobody said whose it was, and the DM asked "whose roll is that?" over and over
    # while the fight never started. A real table does not collect four numbers at once --
    # the DM goes round the table by name, and then a bare number is never ambiguous.
    nxt = pend[0]
    rest = f" Then, in order: {', '.join(pend[1:])}." if len(pend) > 1 else ""
    return (f"[INITIATIVE] Ask {nxt} for their initiative, BY NAME, and nobody else. Then stop "
            f"and wait. The next bare number anyone says is {nxt}'s — log it with "
            f"[INIT {nxt} <number>]. Never ask 'whose roll is that?': you asked one person, so "
            f"it is theirs. If they name someone else with the number, use that name.{rest}\n")


def _reminder(state) -> str:
    """Repeated on EVERY turn, not just in the system prompt. Over a long session the system
    prompt drifts out of mind and the DM slips back into narrating the players' characters in
    the third person ("Oktah presses close to the door") — the one thing players hate most.
    A short reminder next to the player's own words holds far better."""
    import safety
    names = ", ".join(r["character"] for r in state.roster) or "the party"
    # lines and veils are worthless if they only appear in the system prompt at session start
    # and are forgotten forty turns later, so they ride along on every single turn
    guard = safety.prompt_rule(state)
    guard = guard + "\n" if guard else ""
    # a pending level-up is easy to forget once the scene moves on, and then their sheet and
    # their level quietly disagree for the rest of the session
    if pending := [c for c in state.combatants.values() if c.is_pc and c.pending_level]:
        who = ", ".join(f"{c.name}→{c.pending_level}" for c in pending)
        return (guard + f"[REMEMBER] LEVEL-UP PENDING on D&D Beyond ({who}) — unverified, so their old "
                "numbers still stand. If they say they've done it, run [VERIFY]. Otherwise "
                "remind them once at the next natural pause, not every turn. Talk to "
                f"{names} in SECOND PERSON — 'you'. Never narrate what they do or say. Any "
                "number they give you is already their final total.")
    return (guard + f"[REMEMBER] Talk to {names} in SECOND PERSON — 'you'. Never narrate what they do, "
            "say, or feel; that's theirs. Don't restate their action back at them. No commentary "
            "on their choices or their rolls, no filler openers. Any number they give you is "
            "already their final total. Say what happens, then stop." + NEWLINE + SAY_IT_FIRST)


# Nothing is heard until the first token arrives, so whatever the model does BEFORE that first
# token is dead air at the table -- and measured, that is most of the wait. Same session, same
# state, same question, only the instruction changed: 3277ms median to first token today,
# 1967ms when the reply is told to open with the words rather than work up to them, faster on
# 7 of 10 paired prompts.
#
# Ruling out the things it is NOT is what left this. Prompt size does nothing: a 5,195-character
# state block reaches first token in 1622ms and an EMPTY one in 1636ms. State assembly is 4.7ms.
# Speaker identification is 23ms. Whisper is 263ms. The Claude Code session costs 3.7s to start
# and that is paid once per process, not per turn.
#
# It buys the median, not the tail: mean moved 3299 -> 2938ms and p90 barely moved at all, and
# one of the ten runs was the slowest in the whole set. Deliberation is being moved after the
# first word, not removed, so a turn that genuinely needs to think still takes as long -- it
# just starts talking sooner.
NEWLINE = chr(10)
SAY_IT_FIRST = ("[SAY IT FIRST] Open your reply with the words the players hear. Do not think it "
                "through first and do not lead with an [action] — narrate, then act. The table "
                "is sitting in silence until your first word arrives.")


# "DC 12." spoken aloud tells the player whether to bother before the die lands. The prompt
# says not to and the prompt is not reliable, so the number is also removed on the way to the
# voice: a whole sentence that is only a DC is dropped, and an inline one loses the number.
_DC_ONLY = re.compile(r"^\(?\s*(?:that(?:'s| is)\s+)?(?:a\s+)?dc\s*(?:of\s*)?\d{1,2}\s*\)?[.!]?$", re.I)
_DC_INLINE = re.compile(r"[,;(]?\s*\bdc\s*(?:of\s*)?\d{1,2}\s*\)?\s*[,;]?", re.I)


def _speakable(feedback: list) -> str:
    """A short spoken line for a turn that did real work and never opened its mouth.

    Tool results are written for the model, not the table -- "Round 1. Order: Oktah(11),
    Goblin(9). Up first: Oktah" reads fine in a prompt and badly out loud. Take the part that
    is actually news and say that; fall back to something honest rather than silence.
    """
    text = " ".join(str(f) for f in feedback)
    # A NAME, and nothing else. "(.+?)" up to a full stop looked safe and was not: the feedback
    # from several tools is concatenated, "Up first: River Wraith B" had a dice result sitting
    # right behind it with no full stop between, and the table heard
    # "Order's set. River Wraith B [dice] 1d20+4 = 24 [1d20[20] +4], you're up."
    if m := re.search(r"Up first:\s*([A-Za-z][\w' ]{0,28}?)\s*(?:[.\n]|$|🎲|\[)", text):
        return f"Order's set. {m.group(1).strip()}, you're up."
    if m := re.search(r"Still waiting on\s*([A-Za-z][\w', ]{0,40}?)\s*(?:[.\n]|$)", text):
        return f"Still need initiative from {m.group(1).strip()}."
    if m := re.search(r"🎲[^=]*=\s*(\d+)", text):
        n = int(m.group(1))
        return f"That's {'an' if n in (8, 11, 18) or 80 <= n <= 89 else 'a'} {n}."
    return "Got it."


# A whole reply that is the DM having nothing to say. PASS is the word it is told to use;
# the rest are what it reached for on its own when the room was talking to itself, spoken
# aloud one after another until the table asked why it kept saying "silence".
_NOTHING_TO_SAY = re.compile(
    r"^\(?\s*(?:pass|silence|nothing|no reply|no response|quiet|"
    # "None" reached the table as a spoken word -- an f-string wrapped a Python None and the
    # voice engine read it out. Belt and braces: the source is fixed, and it can never be said.
    r"none|null|undefined|n/?a|"
    r"(?:the )?(?:room|table|air) (?:is|stays|remains) (?:silent|quiet))"
    r"\s*\)?[.!]?\s*$", re.I)


def _sentences(text):
    for s in _SENT.split(text.strip()):
        s = s.strip()
        if not s or _DC_ONLY.match(s) or _NOTHING_TO_SAY.match(s):
            continue
        if _DC_INLINE.search(s):
            s = _DC_INLINE.sub("", s).strip()
            s = re.sub(r"\s{2,}", " ", s)
            if not s or s in ".!?":
                continue
        yield s


# ---------------------------------------------------------------- Claude Code (Pro)
class ClaudeCodeBackend:
    def __init__(self):
        self.proc = None
        self.lock = asyncio.Lock()
        # Set when the subscription says no. Play must not stop because of it -- the local
        # model can carry the table until it resets.
        self.limited = ""

    async def _ensure(self):
        if self.proc and self.proc.returncode is None:
            return
        logger.info("starting warm Claude Code session (pre-warming)…")
        env = dict(os.environ)
        # See config.DM_THINKING_TOKENS. The turn's wait is thinking, not the prompt, and this
        # is the only thing measured so far that shortens the WORST case rather than the median.
        if config.DM_THINKING_TOKENS:
            env["MAX_THINKING_TOKENS"] = str(config.DM_THINKING_TOKENS)
        self.proc = await asyncio.create_subprocess_exec(
            config.CLAUDE_BIN, "-p", "--model", config.DM_CLAUDE_MODEL,
            "--output-format", "stream-json", "--input-format",
            "stream-json", "--verbose", "--append-system-prompt", DM_SYS,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env)  # NOT devnull: this is where limits appear
        self.stderr_tail = []
        asyncio.create_task(self._watch_stderr(self.proc))
        await self._send("The session is starting. Reply with exactly: Ready.")
        await self._drain()                       # eat the slow first turn now, not during play
        logger.info("Claude Code session warm.")

    async def _watch_stderr(self, proc):
        """Keep the last few stderr lines so a silent turn can be explained."""
        try:
            while line := await proc.stderr.readline():
                msg = line.decode(errors="replace").strip()
                if not msg:
                    continue
                self.stderr_tail = (self.stderr_tail + [msg])[-8:]
                if diagnose(msg):
                    logger.warning(f"claude: {msg}")
        except Exception:
            pass

    def why_silent(self) -> str:
        """Player-facing reason the last turn produced nothing, if we can tell."""
        return diagnose("\n".join(getattr(self, "stderr_tail", [])))

    async def _send(self, text):
        line = json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"
        self.proc.stdin.write(line.encode())
        await self.proc.stdin.drain()

    def abandon(self):
        """Throw away the turn in flight because the player has moved on.

        Barge-in used to stop the DM SPEAKING while its generator ran on to the end, and the
        next turn then waited behind it on the lock. With a turn taking ten to thirty seconds
        that is precisely the fault the table reported: nothing, nothing, nothing, and then
        every answer at once, each replying to something said a minute ago. A playtest run
        reproduced it exactly -- four prompts silent, the fifth answering the first.

        Dropping the session costs a couple of seconds to re-warm on the next turn, which is
        far cheaper than the player sitting through an answer they have already replaced.
        """
        if self.proc and self.proc.returncode is None:
            logger.info("abandoning the turn in flight; the player has moved on")
            self._kill()

    def _kill(self):
        """Drop the session. The next turn re-warms in a couple of seconds, which is far
        cheaper than talking to a process whose output no longer lines up with our requests."""
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    async def _drain(self, timeout=180):
        while True:
            raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
            if not raw or json.loads(raw.decode() or "{}").get("type") == "result":
                return

    async def _turn(self, text, state, rounds=2, spoke=None, collected=None):
        await self._send(text)
        spoke = spoke if spoke is not None else []    # has anything been said aloud yet?
        feedback, linebuf, seen = [], "", ""
        if collected is None:
            collected = []
        while True:
            raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=TURN_TIMEOUT)
            if not raw:
                break
            try:
                ev = json.loads(raw.decode())
            except Exception:
                continue
            t = ev.get("type")
            if t == "assistant":
                txt = "".join(b.get("text", "") for b in ev.get("message", {}).get("content", [])
                              if b.get("type") == "text")
                delta = txt[len(seen):] if txt.startswith(seen) else txt   # cumulative or delta
                seen = txt if txt.startswith(seen) else seen + txt
                linebuf += delta
                seg, linebuf = self._next_segment(linebuf, first=not spoke)  # stream by clause/sentence
                while seg is not None:
                    async for out in self._line(seg, state, feedback):
                        spoke.append(out)
                        yield out
                    seg, linebuf = self._next_segment(linebuf, first=not spoke)
            elif t == "result":
                # SAY HOW BIG THE CONVERSATION HAS GOT. The CLI reports this on every turn and
                # we were throwing it away, so "is the context large?" could only be guessed at
                # -- and a session's accumulated conversation is the one cost that grows all
                # evening while every other number stays flat.
                u = ev.get("usage") or {}
                cached = u.get("cache_read_input_tokens") or 0
                fresh = u.get("input_tokens") or 0
                if cached or fresh:
                    logger.info(f"context: {cached:,} cached + {fresh:,} fresh in, "
                                f"{u.get('output_tokens') or 0:,} out, "
                                f"turn {ev.get('num_turns')}, "
                                f"${ev.get('total_cost_usd') or 0:.4f}")
                async for out in self._line(linebuf, state, feedback):     # flush the tail
                    spoke.append(out)
                    yield out
                break
        collected.extend(feedback)
        if feedback and rounds > 1:               # let the DM react to real roll/lookup results
            # Two failure modes came out of this round-trip and both are about what the model
            # can see. It could not see what it had ALREADY said, so it said it again -- "Combat
            # starts now. Roll initiative. Roll initiative, Tester." And it was not told that
            # this round is its last chance to speak, so a turn could end having emitted only
            # actions, leaving a bare "That's a 23." from the fallback as the whole answer.
            already = " ".join(spoke[-6:]).strip()
            nudge = "[RESULTS] These came back from the dice and the game state:\n" \
                    + "\n".join(feedback) + "\n"
            if already:
                nudge += (f"[YOU HAVE ALREADY SAID THIS ALOUD THIS TURN] {already}\n"
                          "Do not repeat any of it, do not rephrase it, do not say the same "
                          "handover twice.\n")
            nudge += ("[NOW] Say what these results MEAN in the fiction, in one or two short "
                      "sentences, and then stop. This is the last thing you get to say this "
                      "turn, so if it needs saying, say it now. Never read a die roll, an "
                      "armour class or a difficulty class out loud.")
            async for out in self._turn(nudge, state, rounds - 1, spoke, collected):
                yield out

    @staticmethod
    def _next_segment(buf, first=False):
        """Pull one complete unit off the front of buf: a full [action], or a full narration
        sentence. Returns (segment, rest) or (None, buf) if nothing is complete yet."""
        s = buf.lstrip()
        if not s:
            return None, ""
        if s[0] == "[":                              # a bracketed action -> need the closing ]
            i = s.find("]")
            if i == -1:
                return None, buf
            rest = s[i + 1:].lstrip("\n \t")
            return s[:i + 1], rest
        # narration: emit up to a sentence end or the start of an action, whichever comes first
        m = re.search(r"(?<=[.!?])\s|\n|\[", s)
        if m:
            cut = m.start()
            seg = s[:cut].strip()
            rest = s[cut:] if s[cut] == "[" else s[cut:].lstrip()
            return (seg or None), rest
        if first:
            # Nothing has been SAID yet and the player is sitting in silence. Waiting for a
            # full stop means a long opening sentence is seconds of dead air, so break at the
            # first clause instead and let the voice start there.
            if (c := _CLAUSE.search(s, 25)):
                return s[:c.start() + 1].strip(), s[c.end():]
        return None, buf

    async def _line(self, line, state, feedback):
        line = line.strip()
        if not line:
            return
        # "You've hit your session limit - resets 4:30pm" arrives as an ordinary assistant
        # message, so it went straight to the voice and the DM said it aloud on every single
        # turn. It is not narration and the table should never hear it; it is a fact about the
        # subscription, and what it really means is: use the other brain.
        if diagnose(line):
            self.limited = line
            logger.warning(f"claude unavailable: {line}")
            return
        if action_mod.is_action_line(line):
            _, fb = action_mod.run_action(line, state)
            if fb:
                feedback.append(fb)
            return
        # The server decides whether this is a fight, because the model demonstrably does
        # not: it has narrated three whole combats without emitting a single action.
        import combat as combat_mod
        if (fb := combat_mod.sniff(line, state)):
            feedback.append(fb)
        for s in _sentences(line):                # narration -> speak
            yield s

    async def stream_dm(self, player_text, state, director, history, combat_brief=None):
        async with self.lock:
            try:
                await self._ensure()
                prepared = director.pull_prepared() if director else ""
                msg = f"[STATE]\n{state.summary()}\n"
                if prepared:
                    msg += f"[PREPARED]\n{prepared}\n"
                if getattr(state, "fresh", False):
                    # The continuity block used to be built and then thrown away, so every
                    # session started somewhere new with a history nobody recognised.
                    state.fresh = False
                    import campaign as campaign_mod
                    msg = f"[CONTINUITY]\n{campaign_mod.memory_brief(state)}\n\n" + msg
                if getattr(state, "resumed", False):
                    # We were mid-scene and the process restarted; the model has no memory of
                    # the exchange it was in. Hand it back rather than starting cold.
                    state.resumed = False
                    import transcript
                    if tail := transcript.recent(state.campaign, 8):
                        msg += ("[YOU WERE MID-SCENE — this just happened, continue from it, "
                                "do NOT greet them or restate it]\n" + tail + "\n")
                # The monsters act BEFORE the model is asked anything. Leaving this to a
                # [NEXTTURN] the DM had to remember cost the player fifty-eight seconds of
                # silence and a spoken prompt to make a wraith take its own turn.
                #
                # HybridBackend resolves it before picking a brain, so it arrives already done;
                # only compute it here when nobody else did, or the fight advances twice.
                if combat_brief is None:
                    import combat as combat_mod
                    combat_brief = combat_mod.brief(state, player_text)
                msg += combat_brief or ""
                msg += _waiting_on_initiative(state)
                msg += f"[PLAYER] {player_text}\n{_reminder(state)}"
                with telemetry.Turn("claude_code", config.DM_CLAUDE_MODEL,
                                    getattr(state, "campaign", ""), route="claude_code") as tel:
                    tel.prompt(system=DM_SYS, message=msg, state=state.summary(),
                               player=player_text, prepared=prepared)
                    said, acted = 0, []
                    async for s in self._turn(msg, state, collected=acted):
                        said += 1
                        tel.first_token()
                        tel.chunk(s)
                        yield s
                    if said == 0 and acted:
                        # The turn DID something -- took an initiative, ran a roll -- and said
                        # nothing, so the table saw silence and assumed it had broken. "I
                        # rolled 11." landed exactly here: the number was recorded, no word
                        # came back, and the player repeated themselves fourteen seconds later.
                        line = _speakable(acted)
                        logger.info(f"actions ran but nothing was said; speaking {line!r}")
                        tel.first_token(); tel.chunk(line)
                        yield line
            except (asyncio.CancelledError, GeneratorExit):
                # Abandoned mid-response — a disconnect, or a turn cancelled under us. The
                # session is one long-lived process shared by every future turn, so leaving
                # unread output in the pipe desyncs it permanently: every later turn then
                # reads the tail of this one and hangs. Drop it instead.
                logger.info("claude_code turn abandoned; dropping the session to resync")
                self._kill()
                raise
            except Exception as e:
                logger.warning(f"claude_code turn failed ({e}); restarting session")
                self._kill()
                yield "The DM pauses a moment, gathering the threads of fate."


# ---------------------------------------------------------------- Anthropic API
class ApiBackend:
    def __init__(self):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            AsyncAnthropic = None
        self.client = (AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
                       if AsyncAnthropic and config.ANTHROPIC_API_KEY else None)

    async def stream_dm(self, player_text, state, director, history):
        if self.client is None:
            yield "Set ANTHROPIC_API_KEY (or use the Claude Code backend) to wake the DM."
            return
        prepared = director.pull_prepared() if director else ""
        system = f"{DM_SYS_API}\n\n--- GAME STATE ---\n{state.summary()}"
        if prepared:
            system += f"\n\n--- {prepared} ---"
        history.append({"role": "user", "content": player_text})
        history[:] = history[-40:]
        for _ in range(6):
            msg = await self.client.messages.create(
                model=config.DM_MODEL, max_tokens=500, system=system,
                tools=tool_mod.TOOLS, messages=history)
            history.append({"role": "assistant", "content": msg.content})
            results = []
            for block in msg.content:
                if block.type == "text":
                    for s in _sentences(block.text):
                        yield s
                elif block.type == "tool_use":
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": tool_mod.dispatch(block.name, block.input or {}, state)})
            if results:
                history.append({"role": "user", "content": results})
                continue
            return


DM_SYS_API = ("You are a Dungeon Master running a live tabletop RPG by VOICE. Keep spoken "
              "narration to 1-3 short vivid sentences; stay in character; never say you are an "
              "AI. Use the TOOLS for every mechanic (apply_damage, start_combat, spawn_monster, "
              "lookup_rule, import_character, …) — never invent HP or results.\n"
              + TABLE_RULE + "\n" + DND_RULE + "\n" + CRAFT_RULE + "\n" + CONVERSATION_RULE + "\n" + REVEAL_RULE +
              "\n" + PLAYER_RULE + "\n" + ANSWER_RULE + "\n" + DICE_RULE +
              " Use roll_dice ONLY for your own monsters/NPCs/environment, never for players.\n"
              "Your words are spoken aloud: no markdown, no asterisks, no lists. When an NPC "
              'speaks, write  Name: "the words"  on its own line; the name is a voice cue and is '
              "never read aloud, so do not announce who is speaking.")

# ---------------------------------------------------------------- Ollama (local, instant)
STORY_SYS = (
    "You are a Dungeon Master running a live tabletop RPG by VOICE. Every word you write is "
    "spoken aloud.\n"
    "- ANSWER WHAT THEY ACTUALLY SAID. If they asked a question, answer it. If they asked "
    "whether they can do something, the answer is almost always yes -- say so and say what "
    "happens. Do not reply with scenery to a question about what is possible.\n"
    "- THE SCENE YOU ARE GIVEN IS TRUE. Do not contradict it, do not move anyone, do not "
    "change what they are doing. If it says three figures are standing and blocking the way, "
    "they are standing and blocking the way.\n"
    "- Reply in ONE or TWO short sentences. Three at most, and only when they have just walked "
    "somewhere new. The player is sitting in silence waiting for you.\n"
    "- Second person. Never narrate what the player's character does, says, decides or feels; "
    "that is theirs. Never restate their action back at them.\n"
    "- No commentary on their choices, no praise, no filler openers ('alright', 'okay', 'so').\n"
    "- Any number the player gives you is their FINAL total. Never add to it, never ask what "
    "they rolled. Never say a difficulty class or an armour class out loud.\n"
    "- Give one or two concrete physical details, then stop. No stacked adjectives, no similes, "
    "no dashes mid-sentence, no 'something' or 'somewhere' when you could name the thing.\n"
    "- When an NPC speaks, put their line on its own line as  Name: \"the words\"  and close "
    "the quote on that same line.\n"
    "- Plain spoken words only: no markdown, no asterisks, no lists, no headings.\n"
    "- End with the ball in their court, but not with the same stock question every time.")

COMBAT_SYS = (
    "You are the Dungeon Master narrating a fight OUT LOUD at the table.\n"
    "Everything under [COMBAT] has ALREADY been rolled and applied by the game. Your only job "
    "is to say what happened, in the fiction, briefly.\n"
    "RULES:\n"
    "- Say every '>' fact, in order, in ONE or TWO short sentences total. Then stop.\n"
    "- Use the damage numbers and hit point totals EXACTLY as given. Never change one, never "
    "add one, never invent a number that is not in front of you.\n"
    "- NEVER say an attack roll, an armour class, a difficulty class, or any dice notation. "
    "The player hears what happened to them, not the arithmetic behind it.\n"
    "- A miss is the world reacting, not a failure announcement: say what the blow hit instead.\n"
    "- MOVING IS NOT AN ACTION. A character gets their move AND an action on the same turn, "
    "every turn. Never tell a player that stepping back or repositioning used up their action "
    "-- they can walk and then cast or attack in the same turn.\n"
    "- A TURN ENDS WHEN THE PLAYER SAYS SO, and never before. Not when they have attacked, not "
    "when they have moved, not when they ask you a question. Once they have used their action, "
    "ask once -- 'Anything else, or is that your turn?' -- and then WAIT. They end it by saying "
    "something like 'that's my turn', 'that's it', 'I'm done', 'end of my turn' or 'pass'. "
    "Until you hear that, the turn is still theirs and the monsters do not move.\n"
    "- HAND OVER IN PLAIN ENGLISH: 'Marn, you're up.' or 'That's Wraith B.' There is no "
    "initiative token, nothing is 'handed over' as an object, and you never 'hand over the "
    "damage' -- say what happened and whose go it is.\n"
    # No named example here. It used to read "('Oktah, you're up.')" and the model copied the
    # name straight out of it, handing the turn to a character from a different campaign.
    "- If you are told to hand over, do it in the last few words, using the name you are "
    "given and no other. Never use a name that is not in the facts above.\n"
    "- No commentary, no praise, no filler openers, no markdown, no asterisks, no stage "
    "directions. Plain spoken words only.\n"
    "- Second person to the player. Never narrate what their character decides or feels.")

QUICK_SYS = (
    "You are the Dungeon Master answering a QUICK question at the table. The game state is "
    "given to you; the answer is already in it.\n"
    "RULES:\n"
    "- Answer in ONE short sentence. Fewer words is better. 'Plus three.' is a complete answer.\n"
    "- Read the answer off the game state. Never invent a number that isn't there.\n"
    "- Any number the player states is their FINAL total, modifier already included. Never add "
    "to it, never ask what they rolled.\n"
    "- No commentary, no praise, no encouragement, no filler ('alright', 'okay', 'great').\n"
    "- Never advance the story, describe a scene, introduce anything, or ask what they do next. "
    "You are ONLY answering the question asked.\n"
    "- Speak in second person. Never narrate the player's character.\n"
    "- Plain spoken words only: no markdown, no asterisks, no lists.")

# Questions the local model can answer instantly off the game state. Anything that moves the
# world -- an action, a scene, an NPC -- goes to Claude. When in doubt, it goes to Claude.
_QUICK = re.compile(
    r"^\s*(?:what|whats|what's|how much|how many|how far|do i|does my|is my|am i|can i see|"
    r"remind me|whats my|hows my|how's my|whose|who's turn|whos turn)\b.{0,80}$"
    r"|^\s*(?:what|whats|what's)\s+(?:is|are)\s+my\b",
    re.I)
_QUICK_TOPIC = re.compile(
    r"\b(modifier|mod|bonus|ac|armou?r class|hit points?|hp|health|speed|movement|initiative|"
    r"proficienc\w*|saves?|dc|level|spell slots?|conditions?|whose turn|who'?s turn|"
    r"left|passive|inspiration)\b", re.I)
# never fast-path these, even in question form -- they need the world, memory, or a ruling
_NOT_QUICK = re.compile(r"\b(i attack|i cast|i move|i go|i walk|i open|i search|i look|i talk|"
                        r"i say|i ask him|i ask her|i ask them|happens|should i|do you|tell me "
                        r"about|remember when|last time|what do i see|who is|where am i)\b", re.I)


def is_quick_question(text: str) -> bool:
    """Route only self-contained lookups to the fast local model."""
    t = text.strip()
    if len(t) > 110 or _NOT_QUICK.search(t):
        return False
    return bool(_QUICK.match(t)) and bool(_QUICK_TOPIC.search(t))


# A player handing over a number, mid-fight. "I rolled 17." / "14 to hit." / "6 damage."
_A_NUMBER = re.compile(
    r"^\s*(?:(?:i|that|it)(?:'s|s)?\s+)?(?:i\s+)?"
    r"(?:got|rolled|roll|have|get|was|did|deal|dealt|for)?\s*(?:a\s+|an\s+)?"
    r"(?:nat(?:ural)?\s+)?\d{1,3}"
    r"(?:\s*(?:\+|-|plus|minus|and)\s*\d{1,3})?"
    r"\s*(?:to hit|damage|dmg|total|points?(?: of damage)?)?\s*[.!]?\s*$", re.I)
# A plain attack declaration. Deliberately narrow: anything with a plan in it is story.
_PLAIN_ATTACK = re.compile(
    r"^\s*i(?:'ll| will| am going to| wanna| want to)?\s*"
    r"(?:attack|hit|swing at|strike|stab|shoot|slash)\s+"
    r"(?:the\s+|that\s+|it|him|her|them)?[\w' ]{0,24}"
    r"(?:\s+(?:again|with (?:my|the) [\w' ]{1,20}))?\s*[.!]?\s*$", re.I)
# Words that mean the player is doing something the WORLD has to answer, not the dice.
_STORY_TELL = re.compile(
    r"\b(?:say|says|said|ask|asks|tell|tells|talk|speak|shout|whisper|look|search|examine|"
    r"listen|smell|climb|sneak|hide|open|close|push|pull|lift|read|remember|think|persuade|"
    r"convince|threaten|lie|bluff|cast|spell|rest|camp|travel|walk|go|head|leave|enter)\b",
    re.I)


def is_mechanical_turn(text: str, state) -> bool:
    """A turn that is arithmetic wearing a sentence, and only during a fight.

    The narration benchmark is why this is drawn so tightly. On settled facts the local model
    is perfect and 20x faster; on prose it is measurably worse -- vaguer, three distinct
    openers in six replies against Sonnet's five, and it does not call for checks. So it gets
    the numbers and Sonnet keeps the story.
    """
    if not getattr(state, "in_combat", False):
        return False
    t = (text or "").strip()
    if not t or len(t) > 70 or _STORY_TELL.search(t):
        return False
    return bool(_A_NUMBER.match(t) or _PLAIN_ATTACK.match(t))


class OllamaBackend:
    """A small local model for instant answers to mechanical questions. It never advances the
    story -- that stays with Claude -- so its job is narrow enough for a 14B to do reliably."""

    async def warm(self):
        """Load the model into VRAM now, so the first question of the session isn't the slow one."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
                await client.post(f"{config.OLLAMA_URL}/api/chat", json={
                    "model": config.OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}],
                    "stream": False, "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1}})
            logger.info(f"{config.OLLAMA_MODEL} warm (local quick-answer model)")
        except Exception as e:
            logger.warning(f"ollama warmup failed ({e}); quick questions will use Claude")

    async def stream_dm(self, player_text, state, director, history, combat_brief: str = "",
                        story: bool = False):
        import httpx
        if combat_brief:
            # Narrating a fight the ENGINE already settled is the local model's best job: no
            # decisions, no rules, no memory -- just say these numbers in a sentence. Measured
            # on this GPU it answers in 37ms against Claude Code's 10-30 SECONDS, and it does
            # it without the writerly tells (8.4 words a sentence, no em-dashes, no fluff).
            system, message = COMBAT_SYS, (
                f"{combat_brief}\n"
                f"--- who is here ---\n{state.brief()}\n\n"
                f"The player said: {player_text}")
            opts, route = {"temperature": 0.4, "num_predict": 140}, "combat"
        elif story:
            # Carrying the whole table because Claude is unavailable. Worse prose than Sonnet,
            # and far better than a game that has stopped.
            scene = getattr(state, "scene", "") or ""
            system, message = STORY_SYS, (
                f"--- where they are ---\n{getattr(state, 'location', '')}\n{scene}\n\n"
                f"--- who is here ---\n{state.brief()}\n\n"
                f"--- what just happened ---\n" + "\n".join(list(state.log)[-6:]) + "\n\n"
                f"The player said: {player_text}")
            opts, route = {"temperature": 0.6, "num_predict": 160}, "story-local"
        else:
            system, message = QUICK_SYS, (f"--- CHARACTERS ---\n{state.brief()}\n\n"
                                          f"QUESTION: {player_text}")
            opts, route = {"temperature": 0.2, "num_predict": 80}, "quick"
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": message}],
            "stream": True,
            # Ollama evicts an idle model after ~5 minutes, and reloading 14B costs ~9s -- which
            # would make the "fast" path slower than Claude on the first question after a lull.
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": opts,
        }
        buf = ""
        with telemetry.Turn("ollama", config.OLLAMA_MODEL,
                            getattr(state, "campaign", ""), route=route) as tel:
            tel.prompt(system=system, message=message, player=player_text)
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
                async with client.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        ev = json.loads(line)
                        buf += ev.get("message", {}).get("content", "")
                        seg, buf = _split_sentence(buf)
                        while seg:
                            tel.first_token(); tel.chunk(seg)
                            yield seg
                            seg, buf = _split_sentence(buf)
                        if ev.get("done"):
                            break
            if buf.strip():
                tel.first_token(); tel.chunk(buf.strip())
                yield buf.strip()


def _split_sentence(buf):
    m = re.search(r"(?<=[.!?])\s", buf)
    if not m:
        return None, buf
    return buf[:m.start()].strip(), buf[m.end():]


class HybridBackend:
    """Local model for instant lookups, Claude for everything that moves the story.

    The split is deliberately lopsided: the local model only handles self-contained questions
    that are already answered by the game state, because that is the one job a 14B does as well
    as Claude and ~10x faster. Anything ambiguous goes to Claude."""

    def __init__(self):
        self.story = ClaudeCodeBackend()
        self.quick = OllamaBackend()

    async def _ensure(self):
        """Warm BOTH brains before play: a cold 14B costs ~9s to load into VRAM, which would
        land on the first question of the session."""
        await asyncio.gather(self.story._ensure(), self.quick.warm(), return_exceptions=True)

    async def stream_dm(self, player_text, state, director, history):
        if is_quick_question(player_text):
            logger.info(f"routing to {config.OLLAMA_MODEL} (quick): {player_text!r}")
            try:
                async for s in self.quick.stream_dm(player_text, state, director, history):
                    yield s
                return
            except Exception as e:                    # local model down -> Claude still answers
                logger.warning(f"ollama failed ({e}); falling back to Claude")

        # Resolve the monsters HERE, once, before choosing a brain. It has to happen exactly
        # once per turn -- it mutates the fight -- and it is also the thing that decides where
        # the turn should go.
        import combat as combat_mod
        brief = combat_mod.brief(state, player_text)
        # Mid-fight, a player handing over a number or swinging again is arithmetic wearing a
        # sentence. Those are the local model's best case and they are most of what gets said
        # in a fight, so they should not be costing ten seconds of Sonnet each.
        if not brief and config.LOCAL_COMBAT and is_mechanical_turn(player_text, state):
            brief = combat_mod.resolve_player(state, player_text)
        if brief and config.LOCAL_COMBAT:
            # The engine has already settled every number, so this is narration of fixed facts
            # and nothing else. Measured on this GPU: 37ms to first token against 10-30s
            # through Claude Code, with the numbers stated correctly every time.
            logger.info(f"routing combat narration to {config.OLLAMA_MODEL}")
            try:
                said = False
                async for s in self.quick.stream_dm(player_text, state, director, history,
                                                    combat_brief=brief):
                    said = True
                    yield s
                if said:
                    return
                logger.warning("local combat narration said nothing; using Claude")
            except Exception as e:
                logger.warning(f"ollama combat failed ({e}); falling back to Claude")
        # The subscription said no. Carrying on locally is worse prose than Sonnet and far
        # better than a table that has stopped: without this the DM repeated "you've hit your
        # session limit" as narration on every single turn until the reset.
        if self.story.limited:
            logger.info(f"claude limited; running the story on {config.OLLAMA_MODEL}")
            try:
                said = False
                async for s in self.quick.stream_dm(player_text, state, director, history,
                                                    combat_brief=brief, story=True):
                    said = True
                    yield s
                if said:
                    return
            except Exception as e:
                logger.warning(f"local story fallback failed ({e})")

        said = False
        async for s in self.story.stream_dm(player_text, state, director, history,
                                            combat_brief=brief):
            said = True
            yield s
        # THE ENGINE ALREADY MOVED THE FIGHT. brief() resolved every monster turn before the
        # model was asked, and it mutates state -- so if the model then says nothing, those
        # turns still happened and nobody at the table heard them. Measured at a real session:
        # a screen full of "silent turn: no reason reported" while combat sat still, because
        # the model declined to answer and the facts went with it.
        #
        # A turn the model chose not to narrate is fine out of combat -- somebody said
        # something not meant for the DM. Mid-fight it is not: the scouts moved and swung.
        if not said and brief:
            import combat as combat_mod
            if facts := combat_mod.spoken_facts(brief):
                logger.info("model said nothing mid-combat; speaking the engine's own account")
                for line in _sentences(facts):
                    yield line
        # It went quiet AND told us why: fall back rather than leaving the turn empty.
        if not said and self.story.limited:
            logger.info("claude produced nothing and is limited; answering locally")
            async for s in self.quick.stream_dm(player_text, state, director, history,
                                                combat_brief=brief, story=True):
                yield s


_backend = None


def get_backend():
    global _backend
    if _backend is None:
        _backend = {"api": ApiBackend, "hybrid": HybridBackend,
                    "ollama": OllamaBackend}.get(config.DM_BACKEND, ClaudeCodeBackend)()
        logger.info(f"DM backend: {config.DM_BACKEND}")
    return _backend
