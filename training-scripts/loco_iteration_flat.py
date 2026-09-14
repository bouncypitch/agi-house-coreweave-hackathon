"""
Go1 quadruped locomotion training (flat/rough terrain), run as a detached
subprocess on a molab (marimo) GPU sandbox. This is the script that produced
the locomotion demo's checkpoints, including the final "iter5scratch" 200M-step
rough-terrain policy used in the deck (10+m forward walks, full 20s episodes).

Usage: python3 loco_iteration_flat.py <label> <num_timesteps> <restore_params_path|NONE> <entropy_cost> <seed> [cmd_duration_scale]
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, time, pickle, functools, inspect, types, textwrap
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
import mujoco_playground._src.locomotion.go1.joystick as jmod

LABEL = sys.argv[1]
NUM_TIMESTEPS = int(sys.argv[2])
RESTORE_PARAMS_PATH = sys.argv[3]
ENTROPY_COST = float(sys.argv[4])
SEED = int(sys.argv[5])
CMD_DURATION_SCALE = float(sys.argv[6])

# --- monkey-patch command persistence duration (default mean ~5s -> longer) ---
# Playground resamples the joystick command roughly every
# exponential(rng) * 5.0 seconds. For a demo video you want the robot to hold
# one command long enough to show a clean, sustained walk, so we patch the
# hardcoded 5.0 multiplier directly via source-level monkeypatching (no fork
# of the package needed).
reset_src = textwrap.dedent(inspect.getsource(jmod.Joystick.reset))
step_src = textwrap.dedent(inspect.getsource(jmod.Joystick.step))
reset_src2 = reset_src.replace("jax.random.exponential(key1) * 5.0", f"jax.random.exponential(key1) * {CMD_DURATION_SCALE}")
step_src2 = step_src.replace("jax.random.exponential(key2) * 5.0", f"jax.random.exponential(key2) * {CMD_DURATION_SCALE}")
assert reset_src2 != reset_src, "reset patch did not match"
assert step_src2 != step_src, "step patch did not match"

ns = dict(jmod.__dict__)
exec(reset_src2, ns)
exec(step_src2, ns)
jmod.Joystick.reset = ns["reset"]
jmod.Joystick.step = ns["step"]
print(f"[{LABEL}] patched command duration scale to {CMD_DURATION_SCALE}", flush=True)

base_env_cfg = registry.get_default_config("Go1JoystickFlatTerrain")
base_env_cfg.reward_config.scales.tracking_lin_vel = 3.0
base_env_cfg.reward_config.scales.tracking_ang_vel = 1.0
base_env_cfg.reward_config.scales.orientation = -2.0
base_env_cfg.reward_config.scales.stand_still = -2.0

env = registry.load("Go1JoystickFlatTerrain", config=base_env_cfg)
print(f"[{LABEL}] env loaded (patched class), entropy_cost={ENTROPY_COST}", flush=True)

FLOOR_GEOM_ID = 0
RANGES = {
    "floor_friction": (0.4, 1.0),
    "frictionloss_scale": (0.9, 1.1),
    "armature_scale": (1.0, 1.05),
    "mass_scale": (0.9, 1.1),
}

def make_domain_randomize(ranges):
    friction_lo, friction_hi = ranges["floor_friction"]
    frictionloss_lo, frictionloss_hi = ranges["frictionloss_scale"]
    armature_lo, armature_hi = ranges["armature_scale"]
    mass_lo, mass_hi = ranges["mass_scale"]

    def domain_randomize(model, rng):
        @jax.vmap
        def rand_dynamics(rng):
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
                jax.random.uniform(key, minval=friction_lo, maxval=friction_hi)
            )
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
                key, shape=(12,), minval=frictionloss_lo, maxval=frictionloss_hi
            )
            dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)
            rng, key = jax.random.split(rng)
            armature = model.dof_armature[6:] * jax.random.uniform(
                key, shape=(12,), minval=armature_lo, maxval=armature_hi
            )
            dof_armature = model.dof_armature.at[6:].set(armature)
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, shape=(model.nbody,), minval=mass_lo, maxval=mass_hi)
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
            return geom_friction, dof_frictionloss, dof_armature, body_mass

        friction, dof_frictionloss, dof_armature, body_mass = rand_dynamics(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "geom_friction": 0, "dof_frictionloss": 0, "dof_armature": 0, "body_mass": 0,
        })
        model = model.tree_replace({
            "geom_friction": friction, "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature, "body_mass": body_mass,
        })
        return model, in_axes
    return domain_randomize

NUM_ENVS = 4096
rand_rng = jax.random.split(jax.random.PRNGKey(SEED), NUM_ENVS)
randomize_fn = functools.partial(make_domain_randomize(RANGES), rng=rand_rng)
# NOTE: wrap_for_brax_training's randomization_fn is called with only the
# model argument, so `rng` must be pre-baked via functools.partial here.
wrapped_env = mp.wrapper.wrap_for_brax_training(
    env, episode_length=base_env_cfg.episode_length, action_repeat=1, randomization_fn=randomize_fn,
)
print(f"[{LABEL}] env wrapped, num_envs={NUM_ENVS}", flush=True)

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)

if RESTORE_PARAMS_PATH == "NONE":
    restore_params = None
    print(f"[{LABEL}] training from scratch, no restore", flush=True)
else:
    with open(RESTORE_PARAMS_PATH, "rb") as f:
        restore_params_np = pickle.load(f)
    restore_params = jax.tree_util.tree_map(lambda x: jp.asarray(x), restore_params_np)
    print(f"[{LABEL}] restored params from {RESTORE_PARAMS_PATH}", flush=True)

def progress(step, metrics):
    print(f"[{LABEL}] PROGRESS step={step} reward={metrics.get('eval/episode_reward', 'NA')} "
          f"len={metrics.get('eval/avg_episode_length', 'NA')}", flush=True)

t0 = time.time()
make_policy, params, metrics = ppo(
    environment=wrapped_env,
    num_timesteps=NUM_TIMESTEPS,
    num_envs=NUM_ENVS,
    episode_length=base_env_cfg.episode_length,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    batch_size=256,
    learning_rate=3e-4,
    entropy_cost=ENTROPY_COST,
    discounting=0.97,
    normalize_observations=True,
    network_factory=network_factory,
    seed=SEED,
    num_evals=5,
    num_eval_envs=NUM_ENVS,
    wrap_env=False,
    progress_fn=progress,
    restore_params=restore_params,
)
train_time = time.time() - t0
print(f"[{LABEL}] TRAINING DONE in {train_time:.1f}s", flush=True)

params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
save_path = f"/tmp/loco_params_{LABEL}.pkl"
with open(save_path, "wb") as f:
    pickle.dump(params_np, f)
print(f"[{LABEL}] params saved to {save_path}", flush=True)
print(f"[{LABEL}] ITERATION DONE", flush=True)
