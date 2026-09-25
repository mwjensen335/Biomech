"""Sanity tests for dynamics.py against motions with a known analytic answer.
Run:  python test_dynamics.py
"""
import numpy

import dynamics as D


def rot_z(theta):
    c, s = numpy.cos(theta), numpy.sin(theta)
    out = numpy.zeros((len(theta), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def test_free_rigid_body_with_gyroscopic_term():
    fs, w, radius, mass = 360.0, 8.0, 0.5, 2.0
    t = numpy.arange(0, 2.0, 1 / fs)
    # Body spins about lab Z at constant w; its principal axes are tilted 30 deg from Z,
    # so omega is NOT a principal axis and omega x (I omega) != 0.
    tilt = numpy.radians(30)
    r0 = numpy.array([[1, 0, 0], [0, numpy.cos(tilt), -numpy.sin(tilt)], [0, numpy.sin(tilt), numpy.cos(tilt)]])
    rotation = rot_z(w * t) @ r0
    principal = numpy.array([1.0, 2.0, 3.0])
    com = numpy.stack([radius * numpy.cos(w * t), radius * numpy.sin(w * t), numpy.zeros_like(t)], axis=1)

    seg = D.Segment(name="test", mass=mass, proximal=com, distal=com, rotation=rotation,
                    inertia_local=principal, com_frac=0.5)
    seg.compute_kinematics(1 / fs)
    force, moment = D.newton_euler_step(seg, [])

    sl = slice(20, -20)  # ignore edge differencing
    a_expected = -w ** 2 * com
    assert numpy.allclose(seg.acceleration[sl], a_expected[sl], atol=0.5), "COM acceleration"
    assert numpy.allclose(force[sl], mass * (a_expected - D.GRAVITY)[sl], atol=1.0), "force"

    inertia0 = r0 @ numpy.diag(principal) @ r0.T
    omega0 = numpy.array([0.0, 0.0, w])
    gyro0 = numpy.cross(omega0, inertia0 @ omega0)
    expected = (rot_z(w * t) @ gyro0)
    assert numpy.allclose(seg.alpha[sl], 0, atol=1e-2), "alpha should be ~0"
    assert numpy.allclose(moment[sl], expected[sl], atol=1e-2), (moment[100], expected[100])
    assert numpy.linalg.norm(gyro0) > 1.0, "test must exercise a nonzero gyroscopic term"


def test_chain_reaction_is_equal_and_opposite():
    fs = 360.0
    t = numpy.arange(0, 1.0, 1 / fs)
    x = numpy.stack([numpy.sin(3 * t), numpy.zeros_like(t), numpy.cos(2 * t)], axis=1)
    segs = []
    for i, m in enumerate((1.0, 2.0)):
        prox = x + numpy.array([0, 0, 0.3 * (i + 1)])
        dist = prox + numpy.array([0.3, 0, 0])
        seg = D.Segment(name=f"s{i}", mass=m, proximal=prox, distal=dist,
                        rotation=rot_z(0.5 * t), inertia_local=numpy.array([.01, .02, .03]), com_frac=0.5)
        segs.append(seg.compute_kinematics(1 / fs))
    joints = D.solve_chain({s.name: s for s in segs})
    # Summed over the chain, the force at the proximal end must equal sum m (a - g).
    total = sum(s.mass * (s.acceleration - D.GRAVITY) for s in segs)
    assert numpy.allclose(joints["s1"]["force"], total, atol=1e-6)


if __name__ == "__main__":
    test_free_rigid_body_with_gyroscopic_term()
    test_chain_reaction_is_equal_and_opposite()
    print("all dynamics tests passed")
