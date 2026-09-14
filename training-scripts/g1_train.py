"""
G1 humanoid locomotion training -- Phase 0 of the "quadruped follows humanoid"
feature. Trains a standalone G1JoystickFlatTerrain policy to serve as the
frozen, physics-driven "leader" for the follow task.

Deliberately simpler than the Go1 training script: uses the registry's
DEFAULT reward config unmodified (we haven't earned the right to override G1's
own tuning the way we did for Go1 after many iterations), and no domain
randomization (G1 only needs to reliably execute commands on flat ground, not
be terrain-robust -- unlike Go1 it isn't the thing being scored).

Result from the one run of this script: 100.9M steps in ~51.5 minutes,
eval episode length improved 48/1000 -> 733/1000. Multi-seed manual testing
(see g1_metrics_check.py) confirmed genuine, stable forward walking (seed 3:
survived the full 1000/1000 steps, ~10-14m net forward displacement).

Status: the checkpoint this produced (/tmp/g1_flatscratch.pkl) and its
verification render were never transferred off the sandbox before it died
(HTTP 410 -- same failure mode as the earlier catch-project sandbox). This
script is known-good and ready to re-run against a fresh sandbox.

Usage: python3 g1_train.py <label> <num_timesteps> <seed>
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, time, pickle, functools
import numpy as np
import jax
import jax.numpy as jp

def _device_put_replicated_shim(x, devices):
    devices = list(devices)
    n = len(devices)
    return jax.tree_util.tree_map(
        lambda arr: jax.device_put(jp.broadcast_to(arr, (n,) + jp.shape(arr)), devices[0]),
        x,
    )
jax.device_put_replicated = _device_put_replicated_shim

from brax.training.agents.ppo.train import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import mujoco_playground as mp
from mujoco_playground import registry

LABEL = sys.argv[1]
NUM_TIMESTEPS = int(sys.argv[2])
SEED = int(sys.argv[3])

cfg = registry.get_default_config("G1JoystickFlatTerrain")
env = registry.load("G1JoystickFlatTerrain", config=cfg)
print(f"[{LABEL}] env loaded, obs={env.observation_size} action={env.action_size}", flush=True)

NUM_ENVS = 4096
wrapped_env = mp.wrapper.wrap_for_brax_training(
    env, episode_length=cfg.episode_length, action_repeat=1, randomization_fn=None,
)
print(f"[{LABEL}] env wrapped, num_envs={NUM_ENVS}", flush=True)

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)

def progress(step, metrics):
    print(f"[{LABEL}] PROGRESS step={step} reward={metrics.get('eval/episode_reward', 'NA')} "
          f"len={metrics.get('eval/avg_episode_length', 'NA')}", flush=True)

t0 = time.time()
make_policy, params, metrics = ppo(
    environment=wrapped_env,
    num_timesteps=NUM_TIMESTEPS,
    num_envs=NUM_ENVS,
    episode_length=cfg.episode_length,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    batch_size=256,
    learning_rate=3e-4,
    entropy_cost=0.01,
    discounting=0.97,
    normalize_observations=True,
    network_factory=network_factory,
    seed=SEED,
    num_evals=8,
    num_eval_envs=NUM_ENVS,
    wrap_env=False,
    progress_fn=progress,
    restore_params=None,
)
train_time = time.time() - t0
print(f"[{LABEL}] TRAINING DONE in {train_time:.1f}s", flush=True)

params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
save_path = f"/tmp/g1_{LABEL}.pkl"
with open(save_path, "wb") as f:
    pickle.dump(params_np, f)
print(f"[{LABEL}] params saved to {save_path}", flush=True)
print(f"[{LABEL}] ITERATION DONE", flush=True)

# LESSON LEARNED THE HARD WAY (again) on this run: g1_log_flatscratch.txt grew
# to 37GB from MJX's "solver/linesearch iterations limit reached" warning spam
# and very likely contributed to the sandbox dying mid-session. If re-running,
# redirect this warning spam away or periodically truncate the log file --
# don't just let it grow unbounded across a 100M-step run.
