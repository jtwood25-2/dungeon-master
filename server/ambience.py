"""Procedural background ambience — ORIGINAL generated soundscapes only.

No copyrighted music or sound-effect files are used or downloaded. Each mood is synthesized
from filtered noise with numpy, rendered to a short seamless-looping WAV the client plays quietly
under the game. The DM triggers it with [AMBIENCE <mood>].

This was switched off for a while because under a narrator it competed with the voice you
actually wanted to hear — on a phone speaker, anything audible enough to notice was loud enough
to sit on top of the DM. A real speakerphone on the table changes that: there is headroom for a
bed now, provided it stays a bed.

So it is deliberately quieter and simpler than the version that got turned off:

  no tonal drones     every mood used to carry a sine at 44-98Hz. A held tone is the part the
                      ear locks onto and stops being able to ignore, and it is also the part
                      that fights a voice in the same low range. Beds are filtered noise only.
  four moods, not seven  the old set had a throbbing "battle" texture with its own tremolo,
                      which is scoring rather than ambience. What is left is where the party
                      is — inside, outside, underground, on water — because that is what a
                      room actually sounds like, and it changes far less often than the mood
                      of a scene does.
  much lower          peak 0.55 in the generator against 0.42 in the client multiplied out to
                      something you noticed. It is now a quiet floor you would have to listen
                      for, which is the only kind that survives a whole session.
  longer loops        12s rather than 6s. The seam is what gives a loop away, and hearing it
                      half as often is most of the difference between a bed and a wallpaper.
"""
import io
import wave

import numpy as np

SR = 22050
_CACHE: dict = {}

# Target LOUDNESS of the bed, as RMS, not peak. Normalising to peak let the moods sit three
# times apart in perceived level -- a tavern at 0.120 rms against rain at 0.038, both peaking
# at the same number -- so switching scenes changed the volume of the room. RMS is what the ear
# tracks for a continuous sound, and a bed that jumps when the scene changes is the one thing
# guaranteed to pull attention to it.
#
# The client multiplies by its own AMBIENCE_VOLUME on top, so the audible result is the PRODUCT
# of the two, which is how this ended up loud with neither number looking wrong alone.
RMS = 0.055
PEAK_CEILING = 0.5          # nothing may spike through the narrator, whatever its average
SECONDS = 12.0


def _lp(x, a=0.03):
    """One-pole low-pass — muffled, distant, and cheap. Vectorised: the loop form took ~200ms
    per second of audio, which is silly for something rendered on a state change."""
    b = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        b[i] = acc
    return b


def _bed(mood: str, n: int, rng) -> np.ndarray:
    """Where the party is, as filtered noise. No held tones — see the module docstring.

    Built from white noise shaped by a low-pass, NOT from a cumulative sum. Brown noise is a
    random walk: it wanders away from zero and puts most of its energy under 30Hz, which a
    speakerphone cannot reproduce and a listener cannot hear, while still eating the headroom
    that the peak ceiling is measured against. A spectrum of the first version peaked at 0Hz
    with 48,000x the mean bin — a quiet bed that measured like a loud one.
    """
    noise = rng.standard_normal(n).astype(np.float32)
    m = (mood or "").lower()
    if any(k in m for k in ("rain", "storm", "sea", "ocean", "ship", "tide", "water", "river")):
        t = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
        swell = 0.65 + 0.35 * np.sin(2 * np.pi * 0.6 * t).astype(np.float32)
        return ((noise - _lp(noise, 0.25)) * swell).astype(np.float32)   # moving water: hiss
    if any(k in m for k in ("tavern", "inn", "market", "town", "village", "crowd", "hall")):
        return _lp(noise, 0.06)                       # a room with people in it, heard through a wall
    if any(k in m for k in ("cave", "dungeon", "crypt", "dark", "mine", "shaft", "tunnel")):
        return _lp(noise, 0.03)                       # enclosed stone, very little top end
    return _lp(noise, 0.10)                           # outside: wind across open ground


def _loop(mood: str, secs: float = SECONDS) -> np.ndarray:
    n = int(SR * secs)
    rng = np.random.default_rng(abs(hash(mood)) % (2**32))
    x = _bed(mood, n, rng)
    # Cross-fade the loop into itself so the seam is a fade rather than a click. Fading to
    # SILENCE at both ends, as this used to, makes the bed pulse once every loop -- which is
    # more noticeable than the seam it was hiding.
    f = int(SR * 1.5)
    head, tail = x[:f].copy(), x[-f:].copy()
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    x[:f] = head * ramp + tail * (1.0 - ramp)
    x = x[:-f]
    # DROP DC. The brown-noise beds are a cumulative sum, which wanders away from zero and
    # leaves a large offset at 0 Hz -- inaudible in itself, and it eats the headroom that the
    # peak ceiling is then measured against, so a quiet bed reads as a loud one. A spectrum of
    # the old loops peaked at 0 Hz with 48,000x the mean bin.
    # High-pass at roughly 70Hz. Everything below it is excursion a small speaker spends and
    # nobody hears, and it is also where a narrator's voice has its body -- so it is the exact
    # band a bed must stay out of.
    x = x - _lp(x, 0.020)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-9
    x = x * (RMS / rms)
    peak = float(np.max(np.abs(x))) or 1e-9
    if peak > PEAK_CEILING:                       # a swell must not become a stab
        x = x * (PEAK_CEILING / peak)
    return x.astype(np.float32)


def wav_bytes(mood: str) -> bytes:
    if mood in _CACHE:
        return _CACHE[mood]
    pcm = (_loop(mood) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    _CACHE[mood] = buf.getvalue()
    return _CACHE[mood]

# ------------------------------------------------------------------ one-shot [SFX]
# The names the DM may fire. Lost when this module was rewritten and caught by the test that
# synthesises every one of them -- which is exactly what that test is for.
SFX = ("blade", "hit", "arrow", "door", "thunder", "roar", "magic", "fire",
       "bell", "coin", "scream", "splash", "crack", "heartbeat", "crit", "death")

# Untouched by the quieter beds above: a one-shot is meant to be heard. These fire on a
# specific moment -- a door, a blade, a fall -- and are gone before they can compete with
# anything. Only the LOOPING bed had to get out of the narrator's way.
def _env(n, attack=0.005, decay=0.4, power=2.0):
    a = max(1, int(SR * attack))
    e = np.concatenate([np.linspace(0, 1, a), np.zeros(max(0, n - a))])
    tail = np.exp(-np.linspace(0, 1, max(1, n - a)) / max(1e-3, decay) * 3) ** power
    e[a:] = tail[:max(0, n - a)]
    return e


def _oneshot(name: str) -> np.ndarray:
    m = (name or "").lower().strip()
    rng = np.random.default_rng(abs(hash(m)) % (2**32))

    def dur(s):
        n = int(SR * s)
        return n, np.linspace(0, s, n, endpoint=False), rng.standard_normal(n)

    if m in ("blade", "sword", "clash", "parry"):
        n, t, z = dur(0.55)                       # metal: bright inharmonic partials + scrape
        x = sum(np.sin(2 * np.pi * f * t) * g for f, g in
                ((2400, .5), (3170, .35), (4310, .25), (5600, .15)))
        x = (x + 0.6 * (z - _lp(z))) * _env(n, 0.002, 0.16, 2.4)
    elif m in ("hit", "thud", "impact", "punch"):
        n, t, z = dur(0.45)                       # body blow: low thump + a little grit
        x = (np.sin(2 * np.pi * 90 * t * np.exp(-3 * t)) + 0.35 * _lp(z, 0.15)) * _env(n, 0.001, 0.13)
    elif m in ("arrow", "whoosh", "swing", "miss"):
        n, t, z = dur(0.4)                        # filtered noise sweeping past
        x = (z - _lp(z, 0.25)) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * 0.8
    elif m in ("door", "creak", "gate"):
        n, t, z = dur(1.6)                        # stick-slip creak, then it settles shut
        creak = np.sin(2 * np.pi * (220 + 90 * np.sin(2 * np.pi * 6.5 * t)) * t) * 0.25
        creak *= (np.linspace(1, 0, n) ** 0.5) * (0.5 + 0.5 * np.sin(2 * np.pi * 11 * t))
        boom = np.sin(2 * np.pi * 60 * t) * _env(n, 0.001, 0.1) * 0.5
        x = creak + np.roll(boom, int(SR * 1.15))
    elif m in ("thunder", "storm", "boom"):
        n, t, z = dur(2.2)                        # crack, then rolling low rumble
        rumble = _lp(np.cumsum(z), 0.004); rumble /= np.max(np.abs(rumble)) + 1e-9
        x = rumble * np.exp(-t / 1.1) + 0.5 * (z - _lp(z)) * _env(n, 0.001, 0.05, 3)
    elif m in ("roar", "growl", "monster", "beast"):
        n, t, z = dur(1.3)                        # rough low voice: detuned saws + noise
        base = 70 + 18 * np.sin(2 * np.pi * 2.2 * t)
        saw = sum(((base * k * t) % 1.0 - 0.5) / k for k in (1, 2, 3))
        x = (saw * 0.7 + 0.4 * _lp(z, 0.08)) * np.sin(np.pi * np.linspace(0, 1, n) ** 0.7)
    elif m in ("magic", "spell", "cast", "shimmer"):
        n, t, z = dur(1.1)                        # rising shimmer
        x = sum(np.sin(2 * np.pi * f * t * (1 + 0.5 * t)) * g for f, g in
                ((523, .4), (784, .3), (1046, .25), (1568, .18)))
        x *= np.linspace(0, 1, n) ** 1.5 * np.exp(-t / 0.9)
    elif m in ("fire", "flame", "burn"):
        n, t, z = dur(1.4)
        x = _lp(z, 0.09) * 0.5 + (z - _lp(z)) * (rng.random(n) < 0.004) * 3.0   # crackle pops
    elif m in ("bell", "chime", "temple"):
        n, t, z = dur(2.4)                        # struck bell: inharmonic, long tail
        x = sum(np.sin(2 * np.pi * f * t) * g * np.exp(-t / d) for f, g, d in
                ((440, .5, 1.8), (1050, .3, 1.1), (1760, .2, .7), (2640, .1, .4)))
    elif m in ("coin", "gold", "treasure", "loot"):
        n, t, z = dur(1.0)                        # a handful of coins, struck at random times
        x = np.zeros(n)
        for _ in range(9):
            at = int(rng.random() * n * 0.7)
            f = 2600 + rng.random() * 2600
            seg = np.sin(2 * np.pi * f * t[:n - at]) * _env(n - at, 0.001, 0.05, 3)
            x[at:] += seg * (0.25 + 0.2 * rng.random())
    elif m in ("scream", "cry", "shriek"):
        n, t, z = dur(1.1)
        f = 700 * np.exp(-1.2 * t) + 300
        x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * 0.5
        x = (x + 0.25 * _lp(z, 0.3)) * np.sin(np.pi * np.linspace(0, 1, n) ** 0.6)
    elif m in ("splash", "water", "dive"):
        n, t, z = dur(1.0)
        x = (z - _lp(z, 0.4)) * _env(n, 0.002, 0.12, 1.6) + 0.4 * _lp(z, 0.02) * np.exp(-t / 0.4)
    elif m in ("crack", "bone", "stone", "break"):
        n, t, z = dur(0.5)
        x = (z * (rng.random(n) < 0.02) * 4.0 + np.sin(2 * np.pi * 160 * t)) * _env(n, 0.001, 0.07, 3)
    elif m in ("heartbeat", "heart", "dread", "tension"):
        n, t, z = dur(1.8)                        # lub-dub, twice
        beat = np.sin(2 * np.pi * 48 * t) * _env(n, 0.004, 0.05, 2)
        x = sum(np.roll(beat, int(SR * d)) * g for d, g in
                ((0, 1), (0.32, .7), (0.95, 1), (1.27, .7)))
    elif m in ("crit", "critical"):
        n, t, z = dur(0.9)                        # blade, then a low hit that lands
        x = np.zeros(n)
        blade = _oneshot("blade")[:n]
        x[:len(blade)] = blade * 0.9
        x += np.roll(np.sin(2 * np.pi * 70 * t) * _env(n, 0.001, 0.2), int(SR * 0.09)) * 0.7
    elif m in ("death", "down", "dying"):
        n, t, z = dur(2.6)                        # a low tone falling away
        x = sum(np.sin(2 * np.pi * f * np.exp(-0.25 * t) * t) * g for f, g in ((110, .5), (165, .3)))
        x *= np.exp(-t / 1.5)
    else:
        n, t, z = dur(0.5)                        # unknown name: a soft neutral tick
        x = np.sin(2 * np.pi * 440 * t) * _env(n, 0.002, 0.1)

    if len(x) < 2:
        return np.zeros(2)
    f = min(int(SR * 0.02), len(x) // 4)          # de-click the ends
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    x /= np.max(np.abs(x)) + 1e-9
    return x * 0.7


def sfx_bytes(name: str) -> bytes:
    key = "sfx:" + (name or "").lower().strip()
    if key in _CACHE:
        return _CACHE[key]
    pcm = (_oneshot(name) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    _CACHE[key] = buf.getvalue()
    return _CACHE[key]
