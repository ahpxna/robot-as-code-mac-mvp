"""Safety monitor: independent checks that can override whatever the skills command.

* bumper contact  -> stop, forbid forward motion for a moment, count the collision
* excessive tilt  -> emergency stop (robot tipping or driving off something)
* battery         -> warn when low, emergency stop when critical (brown-out protection)
"""
from __future__ import annotations

import math


class SafetyMonitor:
    def __init__(self, ctx, *, bumper_threshold_n: float = 4.0, tilt_limit_deg: float = 15.0,
                 low_battery: float = 0.15, critical_battery: float = 0.03):
        self.ctx = ctx
        self.bumper_threshold = bumper_threshold_n
        self.tilt_limit = math.radians(tilt_limit_deg)
        self.low_battery, self.critical_battery = low_battery, critical_battery
        self.collisions = 0
        self.estopped = False
        self.reason = ""
        self.battery_warned = False
        self._bumped = False
        self._tilt_since: float | None = None

    def check(self) -> None:
        ctx, st = self.ctx, self.ctx.state
        if st is None:
            return
        bumped = st.bumper_force > self.bumper_threshold
        if bumped and not self._bumped:
            self.collisions += 1
            ctx.drive.inhibit_forward(ctx.time + 1.0)
            ctx.log(f"SAFETY: bumper contact ({st.bumper_force:.0f} N), forward motion inhibited")
        self._bumped = bumped
        a = st.accel
        norm = float(math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)) or 1.0
        tilt = math.acos(max(-1.0, min(1.0, a[2] / norm)))
        if tilt > self.tilt_limit:
            self._tilt_since = self._tilt_since if self._tilt_since is not None else ctx.time
            if ctx.time - self._tilt_since > 0.4:
                self.trigger(f"tilt {math.degrees(tilt):.0f} deg exceeds limit")
        else:
            self._tilt_since = None
        soc = st.battery.soc
        if soc < self.low_battery and not self.battery_warned:
            self.battery_warned = True
            ctx.log(f"SAFETY: battery low ({soc:.0%}); return to the dock soon")
        if soc < self.critical_battery:
            self.trigger(f"battery critical ({soc:.0%})")

    def trigger(self, reason: str) -> None:
        if self.estopped:
            return
        self.estopped, self.reason = True, reason
        self.ctx.log(f"EMERGENCY STOP: {reason}")
        self.ctx.hal.emergency_stop(True)
        self.ctx.cancel()
