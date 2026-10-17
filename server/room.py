"""Who is at the table, on which phone, playing whom, and which device makes the sound.

Every player having their own phone changes three things that were all fudged before.

WHO SAID IT stops being a guess. Speaker identification from voice was the only answer, and at
a real seven-player session the enrolled profiles sat up to 0.703 similar to one another — so
it correctly declined to name anyone most of the time. A player who has tapped their own
character on their own phone is not a similarity score; it is a fact, and it costs nothing.

WHERE THE SOUND COMES OUT stops being all-or-nothing. One device sits by the speakerphone and
plays the DM; the other six show the map and the sheets and stay silent. Seven phones all
narrating at once is what happens without this, and it is unusable.

WHAT EVERYONE SEES stops depending on who happened to speak. Maps, initiative and sheet updates
used to go to the socket that triggered them, so the player who asked was the only one who saw
the map. They are broadcast now.

The registry is deliberately in memory and rebuilt from connections: a stale seat is worse than
no seat, and a phone that has gone away must stop being part of the table the moment it does.
"""
import json
import logging

logger = logging.getLogger("room")

# campaign -> {sock: seat}
_ROOMS: dict = {}


class Seat:
    """One phone at the table."""
    __slots__ = ("sock", "sid", "character", "player", "speaker", "listens")

    def __init__(self, sock, sid: str):
        self.sock = sock
        self.sid = sid
        self.character = ""      # who they are playing, chosen on their own phone
        self.player = ""         # the human's name, from the roster
        self.speaker = True      # does this device play the DM's voice out loud?
        self.listens = True      # does this device send audio up?

    def as_json(self) -> dict:
        return {"sid": self.sid, "character": self.character, "player": self.player,
                "speaker": self.speaker, "listens": self.listens}


def join(campaign: str, sock, sid: str) -> Seat:
    seat = Seat(sock, sid)
    _ROOMS.setdefault(campaign or "default", {})[sock] = seat
    return seat


def leave(campaign: str, sock) -> None:
    r = _ROOMS.get(campaign or "default")
    if r:
        r.pop(sock, None)
        if not r:
            _ROOMS.pop(campaign or "default", None)


def seats(campaign: str) -> list:
    return list(_ROOMS.get(campaign or "default", {}).values())


def seat_of(campaign: str, sock):
    return _ROOMS.get(campaign or "default", {}).get(sock)


def sit(campaign: str, sock, character: str, roster=None) -> dict:
    """Claim a character on this device. Returns the table as everyone should now see it."""
    s = seat_of(campaign, sock)
    if s is None:
        return table(campaign)
    want = (character or "").strip()
    # One character, one phone. Two people on the same seat means the DM cannot tell them apart
    # again, which is the whole thing this exists to stop -- so the newest claim wins and the
    # older device is told it has been displaced rather than silently shadowed.
    for other in seats(campaign):
        if other is not s and other.character.lower() == want.lower() and want:
            other.character = ""
    s.character = want
    for row in (roster or []):
        if (row.get("character") or "").strip().lower() == want.lower():
            s.player = row.get("player") or ""
            break
    logger.info(f"seat: {s.sid[:8]} is playing {want or '(nobody)'}"
                + (f" ({s.player})" if s.player else ""))
    return table(campaign)


def set_audio(campaign: str, sock, on: bool) -> dict:
    s = seat_of(campaign, sock)
    if s is not None:
        s.speaker = bool(on)
        logger.info(f"seat: {s.sid[:8]} speaker {'on' if on else 'off'}")
    return table(campaign)


def table(campaign: str) -> dict:
    """What every phone should be showing about the table itself."""
    ss = seats(campaign)
    return {"type": "table", "seats": [s.as_json() for s in ss],
            "speakers": sum(1 for s in ss if s.speaker)}


def speaking_for(campaign: str, sock) -> str:
    """The character on the phone that sent this audio, or '' if nobody has claimed one."""
    s = seat_of(campaign, sock)
    return s.character if s else ""


async def broadcast(campaign: str, payload: dict, skip=None, origin=None) -> int:
    """Send to every phone at this table. Never lets one dead socket break the others.

    `origin` is not a nicety. A socket that is not in the registry -- a campaign switched after
    joining, a test harness, a client that connected during a restart -- would otherwise be
    sent nothing at all, and the symptom is a phone that shows a blank screen while the table
    plays on. Whoever triggered this always hears about it.
    """
    body = json.dumps(payload)
    sent, n = set(), 0
    for s in seats(campaign):
        if s.sock is skip:
            continue
        try:
            await s.sock.send_text(body)
            sent.add(id(s.sock)); n += 1
        except Exception:
            pass                      # it has gone; the disconnect handler will clear the seat
    if origin is not None and origin is not skip and id(origin) not in sent:
        try:
            await origin.send_text(body)
            n += 1
        except Exception:
            pass
    return n


async def play(campaign: str, audio: bytes, origin=None) -> int:
    """The DM's voice, to the devices that are meant to make a noise.

    Not a broadcast. Seven phones narrating at once is the failure this prevents, and the
    device that happened to hear the player is rarely the one next to the speakerphone.
    """
    n = 0
    for s in seats(campaign):
        if not s.speaker:
            continue
        try:
            await s.sock.send_bytes(audio)
            n += 1
        except Exception:
            pass
    if n == 0 and origin is not None:
        # Everyone muted themselves. Better the asking phone speaks than the table hears
        # nothing at all and concludes the DM has stopped working.
        try:
            await origin.send_bytes(audio)
            n = 1
        except Exception:
            pass
    return n
