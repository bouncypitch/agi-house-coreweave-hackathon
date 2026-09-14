"""
Renders a trained Go1 checkpoint executing a fixed joystick command, and
reports the actual distance walked. This is the final, correct version after
finding the root cause of an hours-long debugging mystery: every earlier
diagnostic script reconstructed the policy network WITHOUT
`preprocess_observations_fn=running_statistics.normalize`, which defaults to
an identity function and silently discards the observation-normalization
statistics baked into the checkpoint -- making a perfectly good trained policy
look like it falls in 1-2 seconds. Passing that one argument turned every
checkpoint's measured behavior from "falls immediately" into "walks 5-16+
meters, often the full episode."

Also folds in a second lesson: Go1's joystick command is HEADING-RELATIVE, not
world-frame, so a constant "forward" command produces a different world-frame
trajectory depending on each episode's random spawn yaw. Test multiple seeds
and pick a genuinely forward-looking one for a demo video, don't assume seed 0
is representative.

Usage: python3 loco_render_forward_final.py <label> <params_path> <seed_idx>
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

env_name = "Go1JoystickRoughTerrain"
base_env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=base_env_cfg)

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    preprocess_observations_fn=running_statistics.normalize,  # <-- the fix
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
    # Command lives at the LAST 3 entries of Go1's state/privileged_state.
    so = state.obs["state"].at[-3:].set(cmd)
    po = state.obs["privileged_state"].at[45:48].set(cmd)
    return state.replace(obs={**state.obs, "state": so, "privileged_state": po}, info={**state.info, "command": cmd})

CMD = jp.array([1.0, 0.0, 0.0])
val_seed = jax.random.PRNGKey(999)
val_keys = jax.random.split(val_seed, 6)
rng = val_keys[SEED_IDX]
state = jit_reset(rng)
state = force(state, CMD)
x0, y0 = float(state.data.qpos[0]), float(state.data.qpos[1])

frames = [state]
for i in range(base_env_cfg.episode_length):
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
print(f"[{LABEL}] episode_length={i+1} distance={dist:.2f}m forward_disp={x1-x0:.2f}m", flush=True)

m = env.mj_model
FLOOR_GEOM_ID = 0
m.geom_rgba[FLOOR_GEOM_ID] = np.array([0.78, 0.72, 0.58, 1.0])
for gi in range(m.ngeom):
    if gi == FLOOR_GEOM_ID:
        continue
    if np.allclose(m.geom_rgba[gi], [0.5, 0.5, 0.5, 1.0]):
        m.geom_rgba[gi] = np.array([0.16, 0.18, 0.22, 1.0])
m.vis.headlight.diffuse[:] = [0.75, 0.75, 0.72]
m.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
m.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

import mediapy as media
imgs = mp.render_array(m, frames, height=1080, width=1440, camera="track")
video_path = f"/tmp/loco_{LABEL}.mp4"
media.write_video(video_path, imgs, fps=int(1.0 / (base_env_cfg.ctrl_dt * 3)), qp=20)
print(f"[{LABEL}] video saved to {video_path}", flush=True)

# NOTE: even with the material/headlight tweaks above, the raw render came out
# badly underexposed (~28/255 average luma). Before using a render, run it
# through a brightness/contrast correction, e.g.:
#   ffmpeg -i in.mp4 -vf "eq=gamma=2.6:contrast=1.25:brightness=0.12:saturation=1.15" out.mp4
# which brought the same footage to ~116/255 average luma -- verified visually
# necessary for legibility on a conference-room projector/TV.
