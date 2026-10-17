"""Decide whether the player has actually finished — and whether they were talking to the DM.

A silence timer alone cannot tell "I open the door." from "I open the door and…". Set it
short and the DM interrupts people mid-thought; set it long and every answer is late. The
transcript shows both failures: "Uh..." got a full narrated response, and a player saying
their own character's name got "that didn't quite come through".

Because speech recognition already runs DURING the end-of-turn pause (see vad.speculate), the
words are available before we have to commit. So the call is made on the text:

    GO     — a complete thought aimed at the DM. Answer now, don't sit out the rest of the pause.
    WAIT   — they're mid-sentence, or thinking. Hold longer; they haven't finished.
    IGNORE — filler, or table talk between players. Don't answer at all, keep listening.

Deterministic and cheap, so it costs nothing and can be tested exhaustively.
"""
import re

GO, WAIT, IGNORE = "go", "wait", "ignore"

# noises people make while thinking. Alone, they are never a turn.
_FILLER_WORD = (r"(?:u+h+|u+m+|e+r+|a+h+|h+m+|m+h?m+|well|so|okay|ok|like|and|but|wait|"
                r"hold on|let me think|thinking|one sec|just a sec|hang on)")
# a whole run of them: "uh, um" and "well... so" are still just thinking noises
_FILLER_ONLY = re.compile(rf"^[\s.,…-]*(?:{_FILLER_WORD}[\s.,…-]*)+$", re.I)

# a sentence that stops on one of these is not finished
_DANGLING = re.compile(
    r"\b(?:and|but|or|so|then|because|cause|if|when|while|with|to|for|at|in|on|into|onto|"
    r"the|a|an|my|his|her|their|its|of|from|toward|towards|about|that|which|who|"
    r"i|we|he|she|they|it|is|are|was|were|am|do|does|did|can|could|would|should|will|"
    r"gonna|going|trying|try|want|wanna|need|let|lets|maybe|actually|like|um|uh)\s*$", re.I)

# said to another player, not to the DM
_TO_OTHERS = re.compile(r"^\s*(?:hey|hold on|wait|no|yeah|yes|ok|okay)[,\s]+(?=\w)", re.I)

_ENDS_CLEAN = re.compile(r"[.!?]\s*$")
# Whisper writes spoken numbers as WORDS as often as digits — the transcript has "Fourteen."
# and "Five.", not 14 and 5 — and a bare roll result must be answered immediately.
_NUM_WORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
             r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
             r"thirty|forty|fifty)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?")
_NUM = rf"(?:\d{{1,3}}|{_NUM_WORD})"
# How a roll result is ACTUALLY said at the table. A bare "Fifteen." was answered instantly
# while "Natural 20." sat out the full hold three times over and answered seven seconds late,
# because two words with no question mark reads as an unfinished sentence.
_NUMBER_ONLY = re.compile(
    rf"^\s*(?:(?:i|that|it)(?:'s|s)?\s+)?(?:i\s+)?(?:got|rolled|roll|have|get|was)?\s*"
    rf"(?:a\s+|an\s+)?(?:nat(?:ural)?\s+)?{_NUM}"
    rf"(?:\s*(?:\+|-|plus|minus|and)\s*{_NUM})?"      # "13 plus 2" — roll and modifier
    rf"\s*(?:total)?\s*[.!]?\s*$", re.I)
_QUESTION = re.compile(
    r"^\s*(?:what|where|when|who|why|how|can|could|do|does|did|is|are|am|should|would|will|"
    r"may|might|whose|which)\b", re.I)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text or "")


# ---- is this even meant for the DM? -----------------------------------------
#
# At a table of seven the microphone hears the room, and most of the room is not talking to
# the Dungeon Master. Session 1 of the group test is the proof: "I'm a suit. It's like two
# minutes. Obi, is this Olaf?", "He's murdered two ideas together.", "They're so happy!" --
# all delivered to the DM as turns, all answered, one after another, with the word "Silence."
#
# A block-list cannot work here; there is no end to the things people say to each other. So
# this is an ALLOW-list: the DM answers when someone declares an action, gives it a number,
# asks it a question about the game, or speaks to it directly. Everything else, it lets go.

# "I attack", "we head north", "I'll cast", "I'm going to look"
_DECLARES = re.compile(
    r"^\s*(?:ok(?:ay)?[,\s]+|so[,\s]+|then[,\s]+|and[,\s]+)?"
    r"(?:i|we|my character)\b[^.!?]{0,12}?\b(?:"
    r"attack|hit|swing|strike|stab|shoot|slash|cast|use|throw|drink|read|open|close|climb|"
    r"jump|swim|run|walk|go|head|move|step|sneak|hide|search|look|examine|inspect|check|"
    r"listen|smell|touch|take|grab|pick|drop|give|hand|show|draw|sheathe|ready|prepare|"
    r"say|tell|ask|shout|whisper|talk|speak|greet|persuade|convince|threaten|lie|bluff|"
    r"intimidate|charm|help|heal|revive|stabilise|stabilize|dodge|dash|disengage|hold|"
    r"wait|rest|camp|sleep|watch|follow|lead|push|pull|lift|tie|cut|burn|light|douse|"
    r"try|attempt|want|wanna|would like|roll|make|turn|end|pass"
    r")\b", re.I)
# addressed to the DM by name or role
_TO_DM = re.compile(r"\b(?:dm|d\.?m\.?|dungeon master|game master|gm)\b", re.I)
# a question that is about the GAME, not about the room
_GAME_Q = re.compile(
    r"\b(?:hit ?points?|hp|ac|armou?r class|initiative|turn|round|spell|slot|cantrip|"
    r"save|saving throw|check|roll|dice|die|advantage|disadvantage|damage|attack|reach|"
    r"range|movement|speed|feet|foot|condition|prone|grappled|inspiration|rest|level|"
    r"proficien|modifier|see|hear|notice|look like|in the room|around (?:me|us)|"
    # whose GO, who's UP, who's NEXT -- the commonest way anyone asks at a table
    r"\bgo\b|\bup\b|next|"
    r"ahead|behind|door|passage|map|where am i|where are we)\b", re.I)

# Addressed to the ROOM rather than to the DM. "did you guys see the game last night" opens
# like a question and mentions "see", so it cleared the gate on both counts -- and it is the
# purest possible example of what the gate exists to drop.
_TO_THE_ROOM = re.compile(
    r"\b(?:you guys|you all|y'?all|you lot|anyone else|any of you|everybody|everyone)\b", re.I)
_QUESTION_OPEN = re.compile(
    # "whose" and "who's" are NOT covered by "who\b" -- there is no word boundary inside them,
    # so "Whose turn is it now?" fell through the gate and was discarded as table talk. That
    # is about the most on-topic question anyone asks at a table.
    r"^\s*(?:what|where|when|whose|who's|whos|who|why|how|which|can|could|may|might|do|does|"
    r"did|is|are|am|was|were|will|would|should|have|has|any)\b", re.I)
# things the table says TO the app rather than to each other
_META = re.compile(
    r"\b(?:recap|open the campaign|begin the campaign|start the session|continue|"
    r"roll(?:ed)? (?:a |an )?\d|initiative is|i got a? ?\d)\b", re.I)


def certainly_for_dm(text: str, party=None, in_combat: bool = False) -> bool:
    """The cheap YES: things that are unmistakably play, decided by pattern in microseconds.

    Its only job is to spare the local gate a round trip on the lines that need no judgement --
    a declared action, a number landing, an end of turn. Everything it is unsure about goes to
    addressed.ask(), which is better at the question than any pattern here.
    """
    import config
    old, config.TABLE_GATE = getattr(config, "TABLE_GATE", "loose"), "strict"
    try:
        return for_the_dm(text, party, in_combat)
    finally:
        config.TABLE_GATE = old


def for_the_dm(text: str, party=None, in_combat: bool = False) -> bool:
    """Should the DM answer this at all, in a room with other people in it?

    Deliberately permissive about ACTIONS and strict about everything else: missing a turn is
    recoverable (they say it again), while answering the room is what makes the DM unusable.
    """
    import config
    t = (text or "").strip()
    if not t:
        return False
    if getattr(config, "TABLE_GATE", "loose") == "off":
        return True
    if _TO_DM.search(t) or _META.search(t):
        return True
    # said to the table, not to the DM -- checked before anything can wave it through
    if _TO_THE_ROOM.search(t):
        return False
    # LOOSE is a deny-list: past the one check above, the DM hears it. The allow-list below was
    # measured at a real seven-player table and rejected almost everything the players said --
    # "They cannot be used to make chemical weapons", "Gus, what do we have on the ship?" -- and
    # a table that is not being heard stops trying long before it works out why.
    if getattr(config, "TABLE_GATE", "loose") != "strict":
        return True
    # ending a turn is the one thing that MUST always get through -- combat stops dead without
    # it, and it rarely looks like an action
    import combat
    if combat.ends_turn(t):
        return True
    if _DECLARES.match(t):
        return True
    if _NUMBER_ONLY.match(t):                      # "Fourteen." -- a roll landing
        return True
    if _QUESTION_OPEN.match(t) and _GAME_Q.search(t):
        return True
    # someone naming a character who is in the party is usually talking about the game
    for name in (party or []):
        first = (name or "").split()[0] if name else ""
        if len(first) > 3 and re.search(rf"\b{re.escape(first)}\b", t, re.I):
            return True
    return False


# Openers that mean "I have not finished the sentence I was already saying".
_CONTINUES = re.compile(
    r"^\s*(?:and|or|but|then|also|plus|so|with|after|before|while|as well as)\b", re.I)

_STOP_PHRASE = re.compile(
    r"^\s*(?:wait|stop|hold on|hang on|hold up|no no|actually|pause|shut up|"
    r"that's wrong|thats wrong|nope)\b", re.I)


def assess(text: str, party: list[str] | None = None, speaking: bool = False,
           speculative: bool = False) -> tuple[str, str]:
    """Return (decision, why). `party` is the other characters/players at the table.

    `speculative` means the transcript came from audio we already know is unfinished -- the
    early-answer path reads the buffer 350ms into a pause, before the pause has proved to be
    the end of anything. That distinction matters because WHISPER PUNCTUATES WHATEVER IT IS
    GIVEN. Handed the first half of a sentence it returns the first half with a full stop on
    the end, and a full stop is the strongest completion signal here.

    A real example from a session: the player said "I think this same sickness is affecting
    our village". The speculative read caught 1.5s of it, Whisper returned "I think the same
    sicknesses are.", the trailing period read as a finished sentence, and the DM answered a
    fragment while the player was still speaking. Without the period the same text is caught
    by the dangling-word check and correctly held.
    """
    t = (text or "").strip()
    if not t:
        return IGNORE, "nothing heard"
    # On the speculative path the trailing period is Whisper's, not the speaker's, so it must
    # not be allowed to mask an unfinished clause. Stripping it outright was too blunt: it also
    # defeated _ENDS_CLEAN, and short but genuinely complete lines like "I'm here to help."
    # started being held. So the period is kept for everything EXCEPT the dangling-word check,
    # which is run against the bare words the audio actually contains.
    bare = re.sub(r"[.!]+\s*$", "", t).strip() if speculative else t
    # Whisper writes a trailing-off sentence as "towards the..." — the ellipsis satisfied
    # "ends cleanly" and beat the dangling-word check, so the DM answered a half-finished
    # thought and told them to carry on. Trailing dots mean the opposite of a full stop.
    trailed = bool(re.search(r"(?:\.{2,}|…)\s*$", t))
    t = re.sub(r"(?:\.{2,}|…)\s*$", "", t).strip() or t

    words = _words(t)
    if _FILLER_ONLY.match(t):
        return IGNORE, "filler only"

    if speaking:
        # The bar to START a turn in silence is not the bar to INTERRUPT one. `speaking` was
        # accepted here and then never used, so a three-word fragment — or the DM's own voice
        # coming back short enough to slip past the echo guard, a bare "Fifteen." included —
        # was enough to cut it off mid-thought. Interrupting must look deliberate.
        if _STOP_PHRASE.match(t):
            return GO, "asked to stop"
        if len(words) < 5:
            return WAIT, "too short to interrupt the DM"
        # A clause that OPENS with a conjunction is the player still finishing their own
        # sentence, not cutting in on the DM's. "Okay, I'll cast Create Bonfire on the
        # rightmost one." / "and take five foot, or ten foot step back." — the second half
        # barged in, killed the answer to the first half, and the turn came back empty. The
        # player then waited fifteen seconds and got the whole fight in one burst.
        if _CONTINUES.match(t):
            return WAIT, "continuing their own sentence"
    if len(words) == 1 and not _NUMBER_ONLY.match(t):
        # one word is almost always a false start; a bare number is a roll result and counts
        return WAIT, "single word"

    # "Thorn, get the door" — addressed to another player, so the DM shouldn't answer.
    #
    # The comma used to be required, and speech-to-text almost never supplies one on a short
    # utterance: "Glemben can you heal me" sailed straight through as a turn for the DM while
    # the identical sentence with a comma was correctly ignored. So the vocative is now
    # recognised by what FOLLOWS the name instead of by punctuation.
    #
    # It has to stay narrow. A name in the third person is narration aimed squarely at the DM
    # ("Glemben is bleeding out"), and so is an action that happens to mention someone ("I
    # follow Glemben into the water"). Only a name in the vocative position — at the very
    # start, followed by second-person address or a bare imperative — means the DM was not the
    # one being spoken to.
    for name in (party or []):
        first = (name or "").split()[0] if name else ""
        if not first or len(first) < 3:
            continue
        lead = rf"^\s*(?:hey\s+|ok(?:ay)?\s+)?{re.escape(first)}\b"
        if re.match(lead + r"\s*[,:]", t, re.I):
            return IGNORE, f"addressed to {first}"
        # no comma: accept only an unambiguous vocative continuation
        if re.match(lead + r"\s+(?:can|could|will|would|do|are|did|have|should)\s+you\b"
                    r"|" + lead + r"\s+you\b"
                    r"|" + lead + r"\s+(?:give|pass|take|grab|help|heal|open|hold|wait|stop|"
                    r"look|come|go|get|move|watch|catch|throw|check)\b", t, re.I):
            return IGNORE, f"addressed to {first}"

    if _NUMBER_ONLY.match(t):
        return GO, "a roll result"                  # "Fourteen." — answer straight away
    if trailed and not _NUMBER_ONLY.match(t):
        return WAIT, "trailed off mid-thought"
    if _DANGLING.search(bare):
        return WAIT, "trailing conjunction"
    if _TO_OTHERS.match(t) and len(words) < 6:
        return WAIT, "sounds like an aside"
    if _ENDS_CLEAN.search(t) and len(words) >= 3:
        return GO, "complete sentence"
    if _QUESTION.match(t) and len(words) >= 3:
        return GO, "a question"
    if len(words) >= 8:
        return GO, "long enough to stand alone"
    return WAIT, "might not be finished"
