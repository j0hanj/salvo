"""motion math. all 2D, meters + seconds, everything vectorized over rows."""

import numpy as np


def pro_nav_turn_rate(m_pos, m_vel, t_pos, t_vel, nav_gain, max_accel, speed):
    """proportional navigation: turn rate ~ nav_gain x closing speed x line-of-sight rotation rate.

    if the line of sight to the target stops rotating you're on a collision course, so pro-nav just
    keeps steering to hold it still. returns turn rate (rad/s), +ve = counterclockwise.
    """
    r = t_pos - m_pos
    v = t_vel - m_vel
    r2 = np.maximum((r * r).sum(-1), 1e-9)
    los_rate = (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) / r2
    closing = -(r * v).sum(-1) / np.sqrt(r2)
    accel = np.clip(nav_gain * closing * los_rate, -max_accel, max_accel)
    return accel / speed


def closest_approach(a0, a1, b0, b1):
    """min distance between two points that each moved in a straight line from *0 to *1 this step.

    checking distance only at the end of the step misses fast stuff that flies straight past each
    other between ticks, so check the whole segment.
    """
    p0 = b0 - a0
    d = (b1 - a1) - p0
    dd = np.maximum((d * d).sum(-1), 1e-12)
    s = np.clip(-(p0 * d).sum(-1) / dd, 0.0, 1.0)
    return np.linalg.norm(p0 + s[:, None] * d, axis=-1)
