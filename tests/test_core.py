import numpy as np
import pytest

from vibtempo import notes, vibrato
from vibtempo.config import load_config
from vibtempo.validate import icc

CFG = load_config(profile="light")


def test_icc_matches_shrout_fleiss_1979():
    # Table 2 of Shrout & Fleiss (1979): 6 targets x 4 judges.
    Y = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    r = icc(Y)
    assert r["ICC1"][0] == pytest.approx(0.17, abs=0.005)
    assert r["ICC2"][0] == pytest.approx(0.29, abs=0.005)
    assert r["ICC3"][0] == pytest.approx(0.71, abs=0.005)
    # CIs as reported by pingouin for the same data.
    assert r["ICC2"][1:] == pytest.approx((0.019, 0.761), abs=0.002)
    assert r["ICC3"][1:] == pytest.approx((0.342, 0.946), abs=0.002)


@pytest.mark.parametrize("rate,depth,dur", [(5.5, 80, 0.4), (6.3, 50, 1.0), (7.4, 120, 2.0)])
def test_fit_recovers_rate_and_depth(rate, depth, dur):
    rng = np.random.default_rng(1)
    t = np.arange(0, dur, 0.01)
    cents = 6000 + 30 * t + depth / 2 * np.sin(2 * np.pi * rate * t + 1.0) + rng.normal(0, 3, len(t))
    m = vibrato.fit(t, cents, CFG["vibrato"])
    assert m["rate_hz"] == pytest.approx(rate, abs=0.1)
    assert m["depth_cents"] == pytest.approx(depth, rel=0.12)
    assert m["confidence"] > 0.8
    assert vibrato.judge(m, CFG["vibrato"]) == ""


def test_fit_rejects_straight_tone():
    rng = np.random.default_rng(2)
    t = np.arange(0, 1.0, 0.01)
    m = vibrato.fit(t, 6000 + rng.normal(0, 3, len(t)), CFG["vibrato"])
    assert vibrato.judge(m, CFG["vibrato"]) != ""


def test_segmentation_splits_steps_and_skips_short_notes():
    hop = 0.01
    t = np.arange(0, 3.0, hop)
    midi = np.where(t < 1.0, 60, np.where(t < 1.15, 62, 64)).astype(float)  # 150 ms middle note
    cents = midi * 100 + 40 * np.sin(2 * np.pi * 6 * t)
    f0 = 440 * 2 ** ((cents - 6900) / 1200)
    c, h = notes.prepare(t, f0, np.ones_like(t), CFG["notes"], CFG["pitch"])
    segs = notes.segment(c, h, CFG["notes"])
    assert len(segs) == 2
    assert segs[0][1] * hop < 1.05 and segs[1][0] * hop > 1.1
