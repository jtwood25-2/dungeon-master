"""DM entry point — delegates to the configured backend (Claude Code on Pro, or the API).
See backend.py. Kept as a thin module so app.py / play.py don't care which brain runs."""
import backend


async def stream_dm(player_text, state, director, history):
    async for sentence in backend.get_backend().stream_dm(player_text, state, director, history):
        yield sentence


async def prewarm():
    """Warm the backend (e.g. boot the Claude Code session) before the first real turn."""
    b = backend.get_backend()
    if hasattr(b, "_ensure"):
        await b._ensure()
