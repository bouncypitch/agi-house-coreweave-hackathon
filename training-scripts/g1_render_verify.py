"""
Renders a trained G1 humanoid checkpoint executing a fixed joystick command,
and reports the actual distance walked -- the Phase 0 verification step for
the "quadruped follows humanoid" feature (a G1 checkpoint must be proven to
walk stably before it's trusted as a frozen leader policy in the merged
scene).

Same two lessons as the Go1 render script apply here:
  1. `preprocess_observations_fn=running_statistics.normalize` is required or
     the policy will look broken even when it isn't.
  2. The joystick command is heading-relative, not world-frame -- test
     multiple seeds and expect different world-frame directions.

One G1-specific detail found by reading g1/joystick.py directly: the command
lives at indices [9:12] of the state/privileged_state vectors (after
linvel[0:3], gyro[3:6], gravity[6:9]), NOT at the end like Go1's [-3:].
Get this wrong and `force()` silently overwrites the wrong slice.

Usage: python3 g1_render_verify.py <label> <params_path> <seed_idx> <vx> <vy> <vyaw>
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, pickle, functools
import jax, jax.numpy as jp
import numpy as np

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

LABEL = sys.argv[1]
PARAMS_PATH = sys.argv[2]
SEED_IDX = int(sys.argv[3])
CMD = jp.array([float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])])

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
with open(PARAMS_PATH, "rb") as f:
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

val_seed = jax.random.PRNGKey(999)
val_keys = jax.random.split(val_seed, 6)
rng = val_keys[SEED_IDX]
state = jit_reset(rng)
state = force(state, CMD)
x0, y0 = float(state.data.qpos[0]), float(state.data.qpos[1])

frames = [state]
for i in range(cfg.episode_length):
    rng, k = jax.random.split(rng)
    action, _ = jit_policy(state.obs, k)
    state = jit_step(state, action)
    state = force(state, CMD)
    if i % 3 == 0:
        frames.append(state)
    if bool(state.done):
        break
x1, y1 = float(state.data.qpos[0]), float(state.data.qpos[1])
dist = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
print(f"[{LABEL}] episode_length={i+1} distance={dist:.2f}m forward_disp={x1-x0:.2f}m cmd={CMD}", flush=True)

m = env.mj_model
FLOOR_GEOM_ID = 0
m.geom_rgba[FLOOR_GEOM_ID] = np.array([0.78, 0.72, 0.58, 1.0])
m.vis.headlight.diffuse[:] = [0.9, 0.9, 0.85]
m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
m.vis.headlight.specular[:] = [0.35, 0.35, 0.35]

import mediapy as media
imgs = mp.render_array(m, frames, height=720, width=960, camera="track")
video_path = f"/tmp/g1_{LABEL}.mp4"
media.write_video(video_path, imgs, fps=int(1.0 / (cfg.ctrl_dt * 3)), qp=20)
print(f"[{LABEL}] video saved to {video_path}", flush=True)
