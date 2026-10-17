"""Per-turn telemetry: what we sent the model, what it cost us in time.

One JSON object per turn appended to state/telemetry.jsonl. The point is to be able to
answer, later and with real data, questions like "does a longer prompt actually slow the
reply down, and by how much" — so record the inputs (prompt size, broken down by the part
of the prompt that produced it) next to the outputs (latency, reply size) on the same row.

Nothing here is on the critical path: recording is best-effort and a failure to log must
never break a turn.
"""
import json
import logging
import time
from pathlib import Path

LOG = Path(__file__).parent.parent / "state" / "telemetry.jsonl"
logger = logging.getLogger("telemetry")

# Rough char->token ratio for English prose. Good enough to spot trends; this is
# deliberately NOT a tokenizer call, which would add latency to every single turn.
CHARS_PER_TOKEN = 4.0


def approx_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN)


class Turn:
    """Times one model turn and writes a row when it closes.

        with telemetry.Turn("claude_code", model, campaign) as t:
            t.prompt(system=DM_SYS, message=msg)
            ... t.first_token() on the first yielded text ...
            ... t.chunk(sentence) per emitted line ...
    """

    def __init__(self, backend: str, model: str = "", campaign: str = "", route: str = ""):
        self.row = {"backend": backend, "model": model, "campaign": campaign, "route": route}
        self.t0 = time.perf_counter()
        self._first = None
        self._reply = []
        self._actions = 0

    # Parts that are established ONCE per session rather than sent every turn. The DM runs a
    # persistent Claude Code session and passes DM_SYS through --append-system-prompt at spawn,
    # so counting it per turn inflated the reported cost of a turn by ~24,000 characters —
    # roughly doubling it — and made every token figure taken off this file wrong.
    ONCE_PER_SESSION = ("system",)
    # Parts that are already inside another part. `state` is embedded in `message` before it is
    # sent; adding both counts the same characters twice.
    EMBEDDED_IN_MESSAGE = ("state", "player", "prepared")

    def prompt(self, **parts: str) -> None:
        """Record prompt size by part, and separate what is actually SENT this turn.

        `chars_sent` is the honest per-turn number and the one to optimise against.
        `chars_prompt` is kept as the total context the model is reasoning over, which is a
        different and also useful quantity — they were conflated, and the difference is most
        of the apparent cost.
        """
        total = sent = 0
        for name, text in parts.items():
            n = len(text or "")
            total += n
            self.row[f"chars_{name}"] = n
            if name not in self.ONCE_PER_SESSION and name not in self.EMBEDDED_IN_MESSAGE:
                sent += n
        self.row["chars_prompt"] = total
        self.row["chars_sent"] = sent
        # not approx_tokens("x" * total): that built a throwaway string the size of the whole
        # prompt (~23KB) every turn, in a module whose whole point is to add no latency
        self.row["tokens_prompt_est"] = int(total / CHARS_PER_TOKEN)
        self.row["tokens_sent_est"] = int(sent / CHARS_PER_TOKEN)

    def first_token(self) -> None:
        if self._first is None:
            self._first = time.perf_counter()
            self.row["ms_first_token"] = round((self._first - self.t0) * 1000)

    def chunk(self, text: str) -> None:
        self._reply.append(text or "")

    def action(self) -> None:
        self._actions += 1

    def fail(self, err: Exception) -> None:
        self.row["error"] = f"{type(err).__name__}: {str(err)[:120]}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and "error" not in self.row:
            self.fail(exc)
        reply = " ".join(self._reply)
        self.row["ms_total"] = round((time.perf_counter() - self.t0) * 1000)
        self.row["chars_reply"] = len(reply)
        self.row["tokens_reply_est"] = approx_tokens(reply)
        self.row["n_sentences"] = len(self._reply)
        self.row["n_actions"] = self._actions
        self.row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write(self.row)
        return False        # never swallow the caller's exception


def write(row: dict) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:                      # telemetry must never break play
        logger.debug(f"telemetry write failed: {e}")
