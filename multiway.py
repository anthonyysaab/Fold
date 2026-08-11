"""Aggressive multiway variant of the Poker Playground policy (SCAFFOLD).

The shipped v3 policy (PurePolicy + DecisionRules) is heads-up-tuned: it folds
almost everything and only commits near the nuts. That wins HU (chipzen, the
dev.fun Eval: +9.95 bb/100) but in a 6-handed Playground table it over-folds,
barely defends its blinds, and plays a very low volume of hands.

MultiwayPolicy loosens that WITHOUT touching the proven base config — it
subclasses PurePolicy and overrides the two levers that gate volume/aggression:

  * the short-handed family selector (open/raise threshold), and
  * the calling discipline (blind defense),

while KEEPING the stack-off gate so it still doesn't stack off light.

This is a first-pass starting point. The knobs in DEFAULT_LOOSE are meant to be
tuned against live Playground S13 sessions (compare chip delta / VPIP vs the
tight version). See the TODO block at the bottom.
"""

from __future__ import annotations

from devfun_poker_playground.pure_model import PurePolicy
from devfun_poker_playground.snapshots import _hero_and_seats, _integer

# --- tuning surface -------------------------------------------------------
# Tight v3 for comparison: aggr floor min(0.72, 0.52 + 0.05*(opp-1)), +0.04
# preflop; call margins {preflop 0, flop .02, turn .05, river .08} plus board
# discount. The values below are deliberately looser to raise VPIP/aggression.
DEFAULT_LOOSE = {
    "aggr_base": 0.42,       # was 0.52 — bet/raise with less equity
    "aggr_per_opp": 0.03,    # was 0.05 — scale up more gently with players
    "aggr_cap": 0.66,        # was 0.72
    # calling / blind-defense margins over pot odds, by street (was much higher)
    "call_margin": {"preflop": -0.02, "flop": 0.0, "turn": 0.02, "river": 0.04},
    # still refuse to commit this fraction of stack below this equity
    "stackoff_fraction": 0.5,
    "stackoff_equity": 0.62,
}


class MultiwayPolicy(PurePolicy):
    """Looser, more aggressive play for multiway (2-6 handed) tables."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loose = dict(DEFAULT_LOOSE)

    def _short_handed_family(self, table, allowed, available, equity, features=None):
        del features  # multiway uses the rules lever, not the tight net head
        _, seats = _hero_and_seats(table)
        opp = max(1, len(seats) - 1)
        cfg = self.loose
        floor = min(cfg["aggr_cap"], cfg["aggr_base"] + cfg["aggr_per_opp"] * (opp - 1))
        # NB: no preflop tightening bump here — we WANT to open more preflop.
        if any(a in available for a in ("bet", "raise")) and equity >= floor:
            return "aggress"
        if "check" in available:
            return "check_call"
        if "call" in available and self._loose_call_ok(table, allowed, equity):
            return "check_call"
        return "fold"

    def _loose_call_ok(self, table, allowed, equity):
        """Wider blind defense than the tight config, but keep the stack-off gate."""
        if equity is None:
            return True
        cfg = self.loose
        margin = cfg["call_margin"].get(self._street(table), 0.04)
        if equity < self._pot_odds(table, allowed) + margin:
            return False
        hero, _ = _hero_and_seats(table)
        stack = _integer(hero.get("stackChips"), "hero stackChips")
        to_call = _integer(allowed.get("callChips", 0), "callChips")
        if to_call >= cfg["stackoff_fraction"] * stack and equity < cfg["stackoff_equity"]:
            return False
        return True


# --- TODO (next session) --------------------------------------------------
# 1. Tune DEFAULT_LOOSE against live Playground S13 (run `live_client.py <comp>
#    --loose`, watch chip delta + how many hands it plays vs the tight version).
# 2. Explicit blind defense: when hero is in the BB facing a min/small raise,
#    call/defend a wide range regardless of the equity floor (position feature
#    is already in the table snapshot).
# 3. Continuation betting: when checked to as the preflop aggressor, bet a
#    sized c-bet at a target frequency instead of checking back marginal equity.
# 4. Light steals: open-raise wider from late position (button/CO) into folded
#    action, using the `position` feature.
# 5. Keep the board-contribution discount active on the AGGRESSION side (don't
#    barrel a hand that's mostly the board) — it's still valuable multiway.
# 6. This override bypasses the tight net head; if a multiway-trained checkpoint
#    ever exists, route to it here via self.table_sizes instead.
