"""The desk the DM speaks from: one voice, one turn at a time, and no queue of stale answers.

Every player's phone holds its own websocket, so four people at a table are four independent
callers into the same DM. Nothing coordinated them. The Claude Code backend is a single
long-lived subprocess behind a single lock, so the turns did not actually overlap -- they
QUEUED, and each one waited out every turn in front of it. Measured across the recorded
sessions: a median turn answers in 2.8 seconds, the ninetieth percentile is 12, and the worst
on record is 82. The long ones are all the same shape, three players speaking inside twenty
seconds during combat.

Queueing is the wrong answer twice over. It is slow, and what arrives at the end of the wait is
worse than nothing: an answer to "I attack the closest one" delivered a minute later lands in a
fight that has moved two turns on, and the table has to work out which question it belongs to.

What a human DM does instead is take them together. So does this. A player who speaks while the
DM has the floor leaves their line at the desk; whoever reaches the desk first collects every
line waiting there and answers all of them in ONE turn. Three players cost one generation, not
three, and the longest anybody waits is a single turn rather than the sum of the ones ahead.

The lock is deliberately per-campaign rather than global: two tables running at once are two
conversations, and neither should be able to make the other wait.
"""
import asyncio
import logging
import time

logger = logging.getLogger("turns")

# Past this, a line is not late -- it is about a moment that has gone. Answering it is worse
# than admitting it was missed, because the table cannot tell a stale answer from a confused
# one. Roughly two slow turns: long enough to survive an ordinary wait, short enough that
# nothing from a previous scene is ever answered as though it just happened.
STALE_SECONDS = 45.0


class Ticket:
    """One player's line, waiting to be spoken for. `taken` once somebody else has it.

    The socket rides along because the answer has to get back to a phone that is not the one
    generating it. One voice for the room, but every screen still shows what was said --
    otherwise being folded into somebody else's turn is indistinguishable from being ignored.
    """
    __slots__ = ("speaker", "text", "at", "taken", "sock")

    def __init__(self, speaker, text, sock=None):
        self.speaker, self.text, self.at, self.taken = speaker, text, time.monotonic(), False
        self.sock = sock

    @property
    def age(self) -> float:
        return time.monotonic() - self.at

    def label(self) -> str:
        return f"[{self.speaker}] {self.text}" if self.speaker else self.text


class Desk:
    """One campaign's turn desk. Hold a ticket, wait for the floor, collect everyone's lines."""

    def __init__(self, campaign: str = ""):
        self.campaign = campaign
        self._lock = asyncio.Lock()
        self._waiting: list[Ticket] = []

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def hold(self, speaker, text, sock=None) -> Ticket:
        t = Ticket(speaker, text, sock)
        self._waiting.append(t)
        return t

    async def floor(self, ticket: Ticket):
        """Wait for the DM's voice. Returns the lines to answer, or None if someone took yours.

        Returning None is the whole economy of this: the player whose line was folded into
        somebody else's turn does not generate a second answer to it.
        """
        await self._lock.acquire()
        if ticket.taken:                              # answered inside another player's turn
            self._lock.release()
            return None
        return self._collect(ticket)

    def _collect(self, ticket: Ticket) -> list[Ticket]:
        """Everything waiting, oldest first, minus anything that has gone stale."""
        mine, stale = [], []
        for t in self._waiting:
            if t.taken:
                continue
            (stale if (t is not ticket and t.age > STALE_SECONDS) else mine).append(t)
        for t in mine + stale:
            t.taken = True
        self._waiting = [t for t in self._waiting if not t.taken]
        if stale:
            logger.info(f"dropping {len(stale)} line(s) older than {STALE_SECONDS:.0f}s: "
                        + "; ".join(repr(t.text[:40]) for t in stale))
        mine.sort(key=lambda t: t.at)
        return mine

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()


def merge(tickets: list[Ticket]) -> str:
    """The prompt text for one turn covering everybody who spoke.

    Said plainly, in speaking order, and labelled as simultaneous -- otherwise the DM answers
    the last one and quietly drops the rest, which is the failure this is supposed to remove.
    """
    if len(tickets) == 1:
        return tickets[0].label()
    lines = "\n".join(f"- {t.label()}" for t in tickets)
    return (f"{len(tickets)} players spoke at once. Answer ALL of them in this one reply, "
            f"in the order given, without asking them to repeat:\n{lines}")


_desks: dict[str, Desk] = {}


def desk(campaign: str = "") -> Desk:
    d = _desks.get(campaign)
    if d is None:
        d = _desks[campaign] = Desk(campaign)
    return d
