"""Voice cast — the narrator and each NPC get a distinct TTS voice.

The DM writes NPC dialogue as its own line `Name: "spoken words"`; everything else is
narration. line_voice() decides which voice speaks a line, assigning a fresh voice to each
new character (persisted per campaign so an NPC keeps its voice all game).
"""
import json
import re
from pathlib import Path

import config

CAST = Path(__file__).parent.parent / "state" / "cast"
NARRATOR = config.TTS_VOICE                      # default narrator voice

# Voice ids are engine-specific, so the pool has to match the engine actually running --
# handing an edge-tts name to the local model just fails.
# Kokoro: mix genders and both accents (a*/b* = American/British) so NPCs stay distinguishable.
# Chosen by measurement, not by taste. tools/voice_audition.py synthesises a fixed narration
# line and a fixed line of NPC dialogue in every voice, transcribes both back with Whisper,
# and scores the word error rate. A voice that slurs or clips its consonants round-trips
# badly, and Whisper listens to all of them identically, so the comparison is fair.
#
# Two things came out of that and both are fixed here:
#   af_nicole scored 0.266 overall and 0.444 on DIALOGUE -- far and away the worst of the 28,
#   and it was in this pool. It is almost certainly why NPCs sounded to the table like a
#   different, rougher system than the narrator.
#   bm_lewis is now the NARRATOR, and it was in this pool too, so an NPC could be handed the
#   DM's own voice. The narrator is excluded from the pool by construction below.
#
# Everything kept scores <= 0.022. am_santa is left out for being a novelty voice rather than
# for any measured fault.
_POOL_ALL = ["bm_george", "af_heart", "am_fenrir", "bf_emma", "bf_alice", "am_onyx",
             "af_sarah", "bf_isabella", "af_sky", "bm_daniel", "af_bella", "af_alloy",
             "af_kore", "am_echo", "bf_lily", "bm_fable", "af_nova", "am_adam"]
# never let an NPC answer in the DM's voice
KOKORO_POOL = [v for v in _POOL_ALL if v != config.TTS_VOICE]
EDGE_POOL = ["en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural",
             "en-US-EmmaMultilingualNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural",
             "en-AU-WilliamNeural", "en-IE-ConnorNeural", "en-US-JennyNeural",
             "en-US-EricNeural", "en-CA-ClaraNeural", "en-GB-ThomasNeural", "en-US-MichelleNeural"]
# NPC voices always come from kokoro — including when the NARRATOR is chatterbox, which has
# one voice. Keying this on == "kokoro" handed edge-tts names to the local model and broke
# every NPC the moment the narrator engine changed.
POOL = EDGE_POOL if config.TTS_ENGINE == "edge" else KOKORO_POOL

# Kokoro voice ids encode gender: a*/b* = American/British, then m/f. Casting a man with a
# woman's voice is the single most jarring thing the DM can do, and it's avoidable.
_MALE = [v for v in POOL if v[1:2] == "m"]
_FEMALE = [v for v in POOL if v[1:2] == "f"]
_HE = re.compile(r"\b(?:he|him|his|man|men|male|father|son|brother|boy|lad|sir|lord|king|"
                 r"husband|uncle|grandfather|monk|priest|smith)\b", re.I)
_SHE = re.compile(r"\b(?:she|her|hers|woman|women|female|mother|daughter|sister|girl|lass|"
                  r"madam|maam|lady|queen|wife|aunt|grandmother|priestess|witch)\b", re.I)


# "A little boy or girl should sound different. An adult should sound different." Age reads
# as strongly as gender in a voice, and casting a child with a grown man's voice is as jarring
# as casting a man with a woman's.
_YOUNG = re.compile(r"\b(?:boy|girl|child|children|kid|lad|lass|young|younger|youth|little|"
                    r"small|infant|toddler|teenage[rd]?|apprentice|urchin|son|daughter)\b", re.I)
_OLD = re.compile(r"\b(?:old|elder|elderly|ancient|aged|grey|gray|grizzled|weathered|crone|"
                  r"grandmother|grandfather|greybeard|venerable|withered|frail)\b", re.I)
# Kokoro voices by apparent age. Lighter/brighter ones read young; deeper, slower ones read old.
# am_puck scored 0.043 and is dropped from the pool; these must stay inside it or
# voice_for falls through to a generic pick and the age hint is silently lost.
_YOUNG_VOICES = [v for v in ("af_sky", "af_bella", "bf_alice", "am_echo")
                 if v in KOKORO_POOL]
_OLD_VOICES = [v for v in ("bm_george", "am_onyx", "am_fenrir", "bm_fable")
               if v in KOKORO_POOL]


def guess_age(text: str) -> str:
    """Returns 'young', 'old', or '' when the description does not say."""
    young, old = len(_YOUNG.findall(text or "")), len(_OLD.findall(text or ""))
    if young and not old:
        return "young"
    if old and not young:
        return "old"
    return ""


def guess_gender(text: str) -> str:
    """Read gender off how the DM describes them. Only decides when it's one-sided."""
    he, she = len(_HE.findall(text or "")), len(_SHE.findall(text or ""))
    if he and not she:
        return "m"
    if she and not he:
        return "f"
    return ""


def _pool_for(gender: str, age: str = "") -> list:
    base = POOL
    if gender == "m" and _MALE:
        base = _MALE
    elif gender == "f" and _FEMALE:
        base = _FEMALE
    if age:
        want = _YOUNG_VOICES if age == "young" else _OLD_VOICES
        # only narrow if the age band still leaves a voice of the right gender
        if narrowed := [v for v in base if v in want]:
            return narrowed
    return base

# A spoken line, in the shapes the DM actually writes it. The name is a CUE for picking the
# voice -- never something to read out loud, since the distinct voice is what tells the player
# who is speaking. The old pattern only accepted `Name: "quoted"`, so every other phrasing
# fell through to the narrator, which then announced the name before the line.
#   Barkeep: "We don't want trouble."   /   **Barkeep:** "..."
_QUOTED = re.compile(r'^\s*[*_]{0,2}\s*([A-Z][\w\'\- ]{1,28}?)\s*[*_]{0,2}\s*:\s*'
                     r'[*_]{0,2}[\"“](.+?)[\"”]')
#   Maren: "Come here.        — the speech runs on into the NEXT fragment
# Narration is split into sentences before it reaches us, so a two-sentence speech arrives
# with the opening quote on one fragment and the closing quote on another. _QUOTED demands
# both, so a speech that ran past one full stop was read entirely in the narrator's voice.
_QUOTED_OPEN = re.compile(r'^\s*[*_]{0,2}\s*([A-Z][\w\'\- ]{1,28}?)\s*[*_]{0,2}\s*:\s*'
                          r'[*_]{0,2}[\"“]([^\"”]+)$')
#   Barkeep: We don't want trouble.      (ambiguous -- only for a character we know)
_BARE = re.compile(r'^\s*[*_]{0,2}\s*([A-Z][\w\'\- ]{1,28}?)\s*[*_]{0,2}\s*:\s*(.+?)\s*$')
#   "Still mine." — a quoted line with no attribution continues the current speaker
_CONTINUES = re.compile(r'^\s*[*_]{0,2}[\"“](.+?)[\"”][\s.,!?]*$')
#   The barkeep says, "We don't want trouble."
_SAYS = re.compile(r'^\s*(?:[Tt]he\s+)?([A-Za-z][\w\'\- ]{1,28}?)\s+'
                   r'(?:says?|replies|answers|growls|snarls|whispers|shouts|calls|mutters|'
                   r'barks|hisses|laughs|sneers|rasps|booms)[^"“]{0,20}[\"“](.+?)[\"”]')

# lines that merely contain a colon are narration, not a character speaking
_NOT_A_NAME = {"you", "the", "it", "they", "he", "she", "there", "then", "but", "and", "now",
               "ahead", "inside", "outside", "behind", "above", "below", "beyond", "suddenly",
               "meanwhile", "roll", "give", "make", "note", "two", "three", "one", "north",
               "south", "east", "west", "left", "right", "wait", "your", "this", "that",
               "everyone", "nothing", "something", "somewhere", "options", "choices"}


def _unquote(text: str) -> str:
    """Strip the quote marks. Sentence-splitting leaves fragments with one dangling quote, and
    a stray character at the start of a line is not something to read aloud."""
    return (text or "").strip().strip('"“”').strip()


def _plausible_name(name: str) -> bool:
    words = name.split()
    return bool(words) and len(words) <= 4 and words[0].lower() not in _NOT_A_NAME


class VoiceCast:
    def __init__(self, campaign="default"):
        self.campaign = campaign
        self.map: dict = {}                      # character(lower) -> voice id
        self.known: set = set()                  # names currently in play (refreshed per turn)
        self.speaking: str = ""                  # who has the floor, for unattributed lines
        self.in_quote: bool = False              # their speech runs on into the next fragment
        self.last_prose: str = ""                # narration before the quote, for gender cues

    @classmethod
    def load(cls, campaign="default"):
        c = cls(campaign)
        p = CAST / f"{campaign}.json"
        if p.exists():
            c.map = json.loads(p.read_text())
        return c

    def save(self):
        CAST.mkdir(parents=True, exist_ok=True)
        (CAST / f"{self.campaign}.json").write_text(json.dumps(self.map))

    def voice_for(self, character: str, gender: str = "", age: str = "") -> str:
        if not character:
            return NARRATOR
        key = character.strip().lower()
        # A cast saved under the other engine holds voice ids this one doesn't have, so
        # re-cast anything unrecognised rather than failing mid-sentence.
        if key not in self.map or self.map[key] not in POOL:
            pool = _pool_for(gender, age)            # nor a child a grown man's
            taken = {v for v in self.map.values()}
            self.map[key] = (next((v for v in pool if v not in taken), None)
                             or next((v for v in POOL if v not in taken), None)
                             or pool[len(self.map) % len(pool)])
            self.save()
        return self.map[key]

    def recast(self, character: str, gender: str) -> str:
        """Move a character onto a voice that matches how they've been described. Called when
        the DM finally says 'the old man' about someone already speaking."""
        key = character.strip().lower()
        pool = _pool_for(gender)
        if not pool or self.map.get(key) in pool:
            return self.map.get(key, NARRATOR)
        taken = {v for k, v in self.map.items() if k != key}
        self.map[key] = next((v for v in pool if v not in taken), pool[0])
        self.save()
        return self.map[key]

    def knows(self, name: str) -> bool:
        return name.strip().lower() in self.map or name.strip().lower() in self.known

    def note_description(self, prose: str):
        """Narration around the dialogue is where gender is actually stated ('the old man
        shakes his head'). Use it to correct anyone already cast the wrong way."""
        for name in list(self.map):
            if re.search(rf"\b{re.escape(name)}\b", prose, re.I):
                if g := guess_gender(prose):
                    self.recast(name, g)
        self.last_prose = prose

    @staticmethod
    def _ends_open(text: str) -> bool:
        """Did this fragment leave a quote hanging open?"""
        return sum(text.count(q) for q in ('"', "“", "”")) % 2 == 1

    def _attribution(self, line: str):
        """(name, spoken, gender, age, ends_open) if this fragment opens an attributed line."""
        for pat in (_QUOTED, _QUOTED_OPEN, _SAYS):
            if (m := pat.match(line)) and _plausible_name(m.group(1)):
                # gender from the attribution itself ("the old man says") or the line before
                g = guess_gender(line[:m.start(2)]) or guess_gender(self.last_prose)
                # age reads as strongly as gender: a child must not get a grown man's voice
                a = guess_age(line[:m.start(2)]) or guess_age(self.last_prose)
                return m.group(1), _unquote(m.group(2)), g, a, self._ends_open(line)
        # unquoted is ambiguous ("Two paths: left and right" is narration), so it only counts
        # for a character we already know is in the scene
        if (m := _BARE.match(line)) and _plausible_name(m.group(1)) and self.knows(m.group(1)):
            return m.group(1), _unquote(m.group(2)), "", "", self._ends_open(line)
        return None

    def line_voice(self, line: str) -> tuple[str, str]:
        """Return (voice_id, spoken_text) for one DM line.

        The hard part is that narration is split into sentences BEFORE it gets here, so a
        two-sentence speech arrives as two fragments and only the first carries the name. The
        old version tracked who spoke last but not whether they were still speaking, so every
        continuation fell back to the narrator — the voice flipping mid-sentence.
        """
        attributed = self._attribution(line)
        # Still inside an open quote from a previous fragment: normally the same character
        # keeps speaking. But the DM routinely forgets the closing quote and slides straight
        # back into narration, leaving this flag stuck on. Two things break that spell:
        #   - a fresh `Name:` attribution (handled below — it always wins), and
        #   - narration that speaks OF the current speaker in the third person
        #     ("Maren draws her hand out"), which a speaker never does about themselves.
        if self.in_quote and self.speaking and attributed is None:
            if re.search(rf"\b{re.escape(self.speaking)}\b", line, re.I):
                self.speaking, self.in_quote = "", False
                self.note_description(line)
                return NARRATOR, line
            if self._ends_open(line):
                self.in_quote = False              # this fragment closes it
            return self.voice_for(self.speaking), _unquote(line)

        if attributed is not None:
            name, spoken, g, a, ends_open = attributed
            self.speaking = name
            self.in_quote = ends_open              # speech continues into the next chunk
            return self.voice_for(name, g, a), spoken
        # a fully quoted line with no attribution continues whoever had the floor
        if (m := _CONTINUES.match(line)) and self.speaking:
            return self.voice_for(self.speaking), _unquote(m.group(1))

        self.speaking, self.in_quote = "", False
        self.note_description(line)
        return NARRATOR, line
