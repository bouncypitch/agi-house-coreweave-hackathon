# Autonomous Robot Learning Loop

**AGI House / CoreWeave Hacks** — robots that detect their own failures, reset themselves, and get better without a human in the loop.

**Live deck:** https://claude.ai/code/artifact/ac803db4-42a9-4199-a41e-3df109ebccbb

## The pitch

Robotics funding hit roughly **$42B in H1 2025** alone — yet robots still aren't part of everyday life. The bottleneck isn't hardware, it's that robots don't learn the way humans do. Three concrete pain points in how robots are trained today:

1. **No self-failure-detection** — when a deployed robot fails, it doesn't know it failed; a human has to notice.
2. **No self-reset** — after a failure, a human has to physically reset the environment before another attempt.
3. **Expensive failure data** — the scenarios a robot most needs to learn from are the most labor-intensive to collect more of.

The vision: show a robot a skill and describe it, decompose it into sub-tasks (via a VLM), and let each sub-task learn itself in a closed loop:

```
evaluate → detect failure → LLM curriculum step → retrain → redeploy → re-measure
```

This repo holds the two demos built end-to-end against that loop, plus a third in progress.

## Demo 1 — Learning to walk on rough terrain

A Unitree Go1 quadruped, trained from scratch via PPO (Brax + MJX, MuJoCo Playground's `Go1JoystickRoughTerrain` task) on a CoreWeave-hosted GPU.

- 200M steps, trained from scratch, with a reward-scale + command-duration curriculum layered on top of Playground's base task (see `training-scripts/loco_iteration_flat.py`)
- Final checkpoint: 10+ meters of verified forward walking, most seeds surviving the full 20-second episode
- Progression videos in `videos/locomotion/`: can't balance → a few unstable steps → walking forward → walking backward, stable

**The big bug, and the actual lesson:** for hours, every manual evaluation script showed the trained policy falling in 1-2 seconds — directly contradicting the training framework's own metrics, which showed ~900+/1000-step episodes. The root cause: `ppo_networks.make_ppo_networks(...)` defaults `preprocess_observations_fn` to an identity function, and every diagnostic script had reconstructed the policy network without passing `preprocess_observations_fn=running_statistics.normalize` — silently discarding the observation-normalization statistics the checkpoint actually contained. The trained policies were fine the whole time; the bug was entirely in the test harness. See `training-scripts/loco_render_forward_final.py` for the fix, and don't skip that argument.

A second, subtler lesson: the joystick command is **heading-relative**, not world-frame, so a constant "forward" command produces a different world-frame trajectory depending on each episode's random spawn yaw. Test multiple seeds before picking one for a demo video.

## Demo 2 — Multi-robot interaction: catching a leaping dog

A humanoid learns to catch a Go1 quadruped that jumps at it — a harder, contact-rich skill, trained with the same loop. Progression in `videos/catch/`: both robots fall on contact → humanoid balances but misses → humanoid catches (shown pre- and post-curriculum-step) → target/goal behavior.

*(The sandbox this project ran on died before its training scripts could be recovered — the videos and pre/post-curriculum comparison survived, the scripts didn't.)*

## Demo 3 (in progress) — Quadruped follows humanoid

The next skill: a Go1 learns to follow a G1 humanoid around, with **both robots fully physics-driven** — the humanoid really walks via its own trained policy and actuators in a shared physics scene, not a scripted puppet. Full design in `plans/follow-the-humanoid-plan.md`.

**Architecture:** hierarchical. The humanoid is a frozen, pretrained `G1JoystickFlatTerrain` policy driven by a scripted command schedule. The quadruped keeps its existing frozen Go1 gait policy for low-level balance/walking, and a new, small "commander" network (the only thing actually trained) maps relative position/heading of the humanoid into the 3D joystick command fed to that frozen gait policy.

**Status:** Phase 0 (train the humanoid's standalone checkpoint) completed successfully — 100.9M steps, eval episode length improved 48→733/1000, and a manual 6-seed check confirmed genuine stable forward walking (best seed: full 1000/1000-step survival, ~10-14m net forward travel — see `training-scripts/g1_metrics_check.py` for the per-seed table). **The checkpoint and its verification video never made it off the sandbox** — it died (HTTP 410) during transfer, the same failure mode that took out the catch-project sandbox earlier. Phase 0's training/render/verification scripts are all in `training-scripts/` and are known-good; re-running them against a fresh sandbox is the next step, followed by Phases 1-2 in the plan (merge the two robots into one physics scene, then PPO-train the commander).

One more finding worth flagging for next time: `g1_log_flatscratch.txt` grew to **37GB** from MJX's solver/linesearch warning spam over a single 100M-step run, and likely contributed to the sandbox's death. Redirect or periodically truncate that log on the next attempt.

## Built with

- **CoreWeave** — every training run's GPU (NVIDIA RTX PRO 6000 Blackwell, 95GB)
- **Molab (marimo)** — the live, GPU-paired notebook the whole loop was orchestrated from
- **Weights & Biases + Weave** — training curve logging and curriculum-step reasoning traces
- **ARIA** — run-history synthesis across iterations
- **Typeface** — noted as a natural next step (auto-generating reports from training telemetry), not yet integrated

## Repo structure

```
deck/                   the pitch deck (HTML source + video-embedding build script)
videos/locomotion/      Demo 1 progression videos, brightness-corrected for projector display
videos/catch/           Demo 2 progression videos
videos/vision/          Disaster-response vision-slide video (concept, not yet built)
training-scripts/       Every training/rendering script recovered from this session, annotated
                         with the bugs each one fixes
plans/                  The approved plan for Demo 3, plus the original pre-hackathon scaffold
```

## Reproducing / continuing this work

All training and rendering ran on a molab (marimo) sandbox with a GPU, using:

```
mujoco, mujoco.mjx, mujoco_playground, jax (GPU build), brax, mediapy, wandb
```

Scripts in `training-scripts/` are written to run as **detached subprocesses**
(`subprocess.Popen(..., stdout=logfile)`), not directly in a notebook kernel —
a failed `import mujoco` (e.g. from a missing system GL library) permanently
breaks that import for the rest of a long-lived kernel process, so all
GPU/import-sensitive work should run in its own process.

To regenerate the deck locally after editing `deck/deck_template.html`:

```
cd deck && python3 build_deck.py
```
