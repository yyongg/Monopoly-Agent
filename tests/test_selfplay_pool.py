"""The opponent pool: what it keeps, and who the agent ends up playing."""

import os

from training.selfplay import OpponentPool, SelfPlayCallback


def _pool(tmp_path, size):
    return SelfPlayCallback(str(tmp_path), snapshot_freq=1, pool_size=size)


def _snaps(tmp_path, steps):
    paths = []
    for s in steps:
        p = tmp_path / f"snapshot_{s:09d}.zip"
        p.write_bytes(b"")
        paths.append(str(p))
    return sorted(paths)


class TestEviction:
    """Pruning to the newest N leaves a sliding window of near-copies of the
    agent's recent self -- it never has to keep beating what it used to be, which
    is how self-play cycles instead of improving."""

    def test_the_earliest_snapshot_is_never_evicted(self, tmp_path):
        cb = _pool(tmp_path, size=3)
        snaps = _snaps(tmp_path, [100, 200, 300, 400, 500, 600])

        dropped = cb._evict(snaps)
        kept = [p for p in snaps if p not in dropped]

        assert len(kept) == 3
        assert snaps[0] in kept, "the anchor must survive"
        assert snaps[-1] in kept, "the newest (strongest) must survive"

    def test_survivors_are_spread_across_training_history(self, tmp_path):
        cb = _pool(tmp_path, size=4)
        steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        snaps = _snaps(tmp_path, steps)

        dropped = cb._evict(snaps)
        kept = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                      for p in snaps if p not in dropped)

        assert len(kept) == 4
        assert kept[0] == 100 and kept[-1] == 1000
        # The old "newest N" rule would have kept 700-1000 -- a single 300-step
        # window. The spread rule reaches back across the whole run.
        assert kept[-1] - kept[0] == 900
        assert max(kept) - min(kept) > 300

    def test_nothing_is_dropped_below_the_cap(self, tmp_path):
        cb = _pool(tmp_path, size=10)
        snaps = _snaps(tmp_path, [100, 200, 300])
        assert cb._evict(snaps) == []


class TestRosters:
    def test_the_fp_trio_stays_a_trio(self, tmp_path):
        """FP is the benchmark the agent is measured on, so when it is drawn it
        must be drawn whole -- sprinkling FP bots seat-by-seat at p=0.3 would
        make a full trio a 2.7% event."""
        pool = OpponentPool(str(tmp_path), baseline_prob=0.0, fp_prob=1.0, seed=0)
        roster = pool()
        assert len(roster) == 3
        assert all(hasattr(bot, "decide") for bot in roster)
        # Three *distinct* profiles, i.e. FP-A/B/C rather than one bot thrice.
        assert len({id(bot) for bot in roster}) == 3

    def test_an_empty_pool_falls_back_to_the_baseline(self, tmp_path):
        pool = OpponentPool(str(tmp_path), baseline_prob=0.0, fp_prob=0.0, seed=0)
        roster = pool()
        assert roster == [None, None, None]   # every seat on the engine baseline
