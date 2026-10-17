"""Keep the test suite out of the real state directory.

Running the tests was writing into `state/transcripts/` and `state/audio/` for real: thirteen
`unittest *.jsonl` transcripts and four empty wavs had accumulated there. That was survivable
while state was only ever read by a human, and stops being survivable now that those exact
directories are the training corpus -- "I attack the ogre" repeated twenty-four times per run
is not a thing any model should learn from, and empty recordings would be segmented and
labelled alongside real ones.

Redirected per-test, so a leak can only ever reach a temp directory.

`game_state.STATE_DIR` is deliberately NOT redirected here: many tests already patch it
themselves, and the characterisation test reads the real transcript history on purpose (by
absolute path, not through this module) to measure how the live rule behaves on real speech.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "server"))


@pytest.fixture(autouse=True)
def _isolate_written_state(tmp_path, monkeypatch):
    import audiolog
    import transcript
    monkeypatch.setattr(audiolog, "ROOT", tmp_path / "audio")
    monkeypatch.setattr(transcript, "DIR", tmp_path / "transcripts")
    yield
