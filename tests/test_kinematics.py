import numpy as np

from sim.kinematics import closest_approach, pro_nav_turn_rate


def fly(m_pos, heading, speed, t_pos, t_vel, steps=4000, dt=0.01):
    """tiny standalone pro-nav loop, returns the closest the missile ever got."""
    best = np.inf
    for _ in range(steps):
        m_vel = speed * np.array([[np.cos(heading), np.sin(heading)]])
        w = pro_nav_turn_rate(m_pos, m_vel, t_pos, t_vel, 3.0, 500.0, speed)[0]
        heading += w * dt
        m_new = m_pos + m_vel * dt
        t_new = t_pos + t_vel * dt
        best = min(best, closest_approach(m_pos, m_new, t_pos, t_new)[0])
        m_pos, t_pos = m_new, t_new
    return best


def test_pro_nav_hits_crossing_target():
    # target crossing left to right, missile launched pointing straight at where it is now (wrong)
    miss = fly(
        np.zeros((1, 2)), np.pi / 2, 300.0, np.array([[-1000.0, 2000.0]]), np.array([[50.0, 0]])
    )
    assert miss < 2.0


def test_no_turn_when_on_collision_course():
    # head-on, line of sight not rotating -> zero turn command
    w = pro_nav_turn_rate(
        np.zeros((1, 2)),
        np.array([[100.0, 0]]),
        np.array([[1000.0, 0]]),
        np.array([[-50.0, 0]]),
        3.0,
        500.0,
        100.0,
    )
    assert abs(w[0]) < 1e-12


def test_closest_approach_catches_fly_through():
    # two points swap places in one step, end-of-step distance is 20 but they actually crossed
    d = closest_approach(
        np.array([[0.0, 0]]), np.array([[10.0, 0]]), np.array([[10.0, 0]]), np.array([[-10.0, 0]])
    )
    assert d[0] < 1e-9


def test_closest_approach_vectorized():
    a0 = np.zeros((3, 2))
    a1 = np.zeros((3, 2))
    b0 = np.array([[5.0, -10], [7.0, 0], [0, 3.0]])
    b1 = np.array([[5.0, 10], [7.0, 0], [0, 4.0]])
    assert np.allclose(closest_approach(a0, a1, b0, b1), [5.0, 7.0, 3.0])
