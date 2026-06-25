"""Bit-exact validation: JAX rsa_bitfaithful trajectory == C++ rsa_trace output."""
import os, subprocess
import numpy as np
import pytest
from rsa_bitfaithful import trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
TRACER = os.path.join(HERE, "rsa_trace")


def _run_tracer(stages, ips, b0, b1, gseed, sseed, replica, N):
    out = subprocess.run(
        [TRACER, str(stages), str(ips), str(b0), str(b1),
         str(gseed), str(sseed), str(replica), str(N)],
        capture_output=True, text=True, check=True).stdout
    rows = [ln.split() for ln in out.strip().splitlines()]
    return np.array([int(r[1]) for r in rows], dtype=np.int64)


def _build_tracer():
    if not os.path.exists(TRACER):
        subprocess.run(["g++", "-O2", "-o", TRACER,
                        os.path.join(HERE, "rsa_trace.cpp")], check=True)


@pytest.mark.parametrize("cfg", [
    # (stages, iters_per_stage, beta_start, beta_end, replica)
    (1, 2000, 4.0, 4.0, 0),     # constant beta
    (4, 500, 0.25, 6.0, 0),     # linear schedule, total=2000
    (8, 400, 0.5, 6.0, 3),      # different replica
])
def test_jax_matches_cpp_tracer_exactly(cfg):
    _build_tracer()
    stages, ips, b0, b1, replica = cfg
    N, gseed, sseed = 2048, 1, 2
    jax_traj = trajectory(gseed, sseed, replica, N, stages, ips, b0, b1)
    cpp_traj = _run_tracer(stages, ips, b0, b1, gseed, sseed, replica, N)
    assert jax_traj.shape == cpp_traj.shape
    assert np.array_equal(jax_traj, cpp_traj), (
        "first diverge at idx %s: jax=%s cpp=%s" % (
            int(np.argmax(jax_traj != cpp_traj)),
            jax_traj[jax_traj != cpp_traj][:1],
            cpp_traj[jax_traj != cpp_traj][:1]))
