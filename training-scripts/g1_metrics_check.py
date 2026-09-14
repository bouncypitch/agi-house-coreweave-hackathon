"""
Quick multi-seed metrics-only check for a trained G1 checkpoint (no video
render -- just episode length / forward displacement / lateral drift per
seed). Used to pick a genuinely forward-walking seed before spending time on
a full-quality render, given the joystick command is heading-relative and
different seeds' spawn yaw produces very different world-frame trajectories.

Result from the one run of this script against /tmp/g1_flatscratch.pkl with a
constant [1,0,0] "forward" command held for the full 1000-step episode:

  seed  episode_length  forward_disp  lateral_drift
  0     368 (fell)      -3.29m        -3.50m
  1     1000            -13.92m       -6.93m
  2     526 (fell)      +6.87m        +2.90m
  3     1000            +14.00m       +7.24m   <- picked for the demo render
  4     1000            +8.60m        +13.95m
  5     829 (fell)      -12.08m       -3.15m

Takeaway: a constant local "forward" command does NOT track a straight
world-frame line over a long episode -- small heading-tracking error
accumulates into real lateral drift even at commanded vyaw=0. Worth
accounting for when designing the Phase 1 scripted leader path (periodic
small vyaw corrections rather than assuming a constant command stays
straight).
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import functools
import jax, jax.numpy as jp
import pickle

def _device_put_replicated_shim(x, devices):
    devices = list(devices)
    n = len(devices)
    return jax.tree_util.tree_map(
        lambda arr: jax.device_put(jp.broadcast_to(arr, (n,) + jp.shape(arr)), devices[0]),
        x,
    )
jax.device_put_replicated = _device_put_replicated_shim

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
import mujoco_playground as mp
from mujoco_playground import registry

env_name = "G1JoystickFlatTerrain"
cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=cfg)

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
with open("/tmp/g1_flatscratch.pkl", "rb") as f:
    params_np = pickle.load(f)
params = jax.tree_util.tree_map(lambda x: jp.asarray(x), params_np)
obs_size = {"state": env.observation_size["state"], "privileged_state": env.observation_size["privileged_state"]}
ppo_net = network_factory(observation_size=obs_size, action_size=env.action_size)
policy_fn = ppo_networks.make_inference_fn(ppo_net)(params, deterministic=True)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_policy = jax.jit(policy_fn)

def force(state, cmd):
    so = state.obs["state"].at[9:12].set(cmd)
    po = state.obs["privileged_state"].at[9:12].set(cmd)
    return state.replace(obs={**state.obs, "state": so, "privileged_state": po}, info={**state.info, "command": cmd})

CMD = jp.array([1.0, 0.0, 0.0])
val_seed = jax.random.PRNGKey(999)
val_keys = jax.random.split(val_seed, 6)

for seed_idx in range(6):
    rng = val_keys[seed_idx]
    state = jit_reset(rng)
    state = force(state, CMD)
    x0, y0 = float(state.data.qpos[0]), float(state.data.qpos[1])
    for i in range(cfg.episode_length):
        rng, k = jax.random.split(rng)
        action, _ = jit_policy(state.obs, k)
        state = jit_step(state, action)
        state = force(state, CMD)
        if bool(state.done):
            break
    x1, y1 = float(state.data.qpos[0]), float(state.data.qpos[1])
    dist = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
    print(f"seed={seed_idx} episode_length={i+1} distance={dist:.2f}m forward_disp={x1-x0:.2f}m lateral={y1-y0:.2f}m", flush=True)
