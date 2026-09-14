# Quadruped-follows-humanoid: build the physics-driven multi-robot follow task

## Context

The hackathon submission is already in (locomotion + catch demos, both trained via an evaluate → detect-failure → LLM-curriculum → retrain loop on a CoreWeave-hosted GPU, molab sandbox `sb-da66c7aabfbe811c`). The user is now continuing the project post-submission and wants to build a third skill: a Go1 quadruped learns to follow a G1 humanoid around, with **both robots fully physics-driven** (the humanoid really walks via its own trained policy and actuators, not a scripted/kinematic puppet) — chosen explicitly over the cheaper kinematic-puppet alternative for authenticity, accepting the extra engineering risk.

This revives the team's original pre-hackathon plan for this exact feature (previously scaffolded as a blank local notebook), but the situation is now completely different: there's a live molab sandbox with real trained checkpoints and hard-won infra patterns from this session, not an empty local file. This plan replaces that stale scaffold.

Confirmed by directly inspecting the live sandbox's installed `mujoco_playground`:
- `Go1JoystickFlatTerrain` / `Go1JoystickRoughTerrain`: obs `{state:(48,), privileged_state:(123,)}`, action `(12,)`. We already have a working frozen checkpoint: `/tmp/loco_params_iter5scratch.pkl` (rough terrain, 200M steps, verified walking 10+m) and `/tmp/loco_params_flatscratch.pkl` (flat terrain).
- `G1JoystickFlatTerrain` / `G1JoystickRoughTerrain` (Unitree G1 humanoid — real registry env, not `H1`): obs `{state:(103,), privileged_state:(216,)}`, action `(29,)`. **No checkpoint exists yet — this must be trained from scratch first.**
- Both robots' scene XMLs live at predictable paths (`mujoco_playground/_src/locomotion/{go1,g1}/xmls/scene_mjx_*_flat_terrain.xml`), each pulling the base robot body from `external_deps/mujoco_menagerie/unitree_{go1,g1}/`. No combined two-robot scene exists in the registry — this has to be built.
- No files from the earlier catch project (also a multi-robot scene) survive — that sandbox died, so there's no prior merged-scene code to reuse; this is being built fresh.

## Architecture

**Hierarchical, two frozen low-level policies + one new trainable high-level policy:**

- **Humanoid (leader):** a frozen, pretrained `G1JoystickFlatTerrain` policy, driven every step by a scripted command schedule (walk forward → turn → speed up → stop, repeating/randomized per episode) — same role the random command sampler plays in the stock task, just deterministic/scripted instead of random. It does not learn; it's a moving, physically-real target.
- **Quadruped (follower), low level:** the existing frozen `loco_params_iter5scratch.pkl` Go1 gait policy — unchanged, already proven. It still consumes a 3D joystick command `[vx, vy, vyaw]` and produces the 12 joint torques, exactly as it does today.
- **Quadruped (follower), high level — the new thing to train:** a small "commander" network. Input: dog proprioception + relative position/heading/velocity of the humanoid. Output: the 3D joystick command fed to the frozen gait policy above. This is the only network that gets PPO-trained, and it's small because the hard part (balance, gait) is already solved.

Both frozen policies run *inside* the environment's `step()` as plain closed-over JAX functions (no gradients flow through them) — only the commander is the RL agent the outer PPO trainer sees. This mirrors how `Joystick.step` already turns a command into torques; we're adding one more hierarchical layer on top for the dog, and reusing the whole stack unchanged for the humanoid.

## Phases

**Phase 0 — Train the humanoid's own locomotion checkpoint.**
Reuse `/tmp/loco_iteration_flat.py` almost as-is, retargeted at `G1JoystickFlatTerrain` (swap the registry name, obs/action dims are read from the env so should mostly fall out automatically; the existing command-duration monkeypatch on `Joystick.reset`/`Joystick.step` needs verifying against G1's `joystick.py`, since it's a separate module from Go1's even though the pattern is likely shared). Train from scratch, same detached-subprocess pattern, save to `/tmp/g1_flatscratch.pkl`. Verify with a render exactly like the Go1 checks (does it walk, for how long, how far) before moving on — do not build the merged scene against an unverified humanoid checkpoint.

**Phase 1 — Build and de-risk the merged physics scene, with a hand-coded commander first.**
1. Write a scene-merge script (`/tmp/follow_scene_build.py`) that loads both scene MJCFs, prefixes one robot's names to avoid collisions, attaches the humanoid's body subtree into the dog's world at a spawn offset (e.g. 2–3m ahead), keeps one shared floor (the Go1 flat-terrain floor; drop G1's), and concatenates actuator lists (dog's 12 first, then G1's 29 — 41 total) into one combined MJCF.
2. Write `/tmp/follow_env.py`: a custom `MjxEnv` subclass that on `reset` places both robots and initializes the humanoid's scripted path/command schedule; on `step` splits/derives the two frozen policies' torques (commander's 3D output → frozen Go1 policy → 12 torques; scripted command → frozen G1 policy → 29 torques), concatenates to 41 actuator inputs, steps physics once, and returns an observation/reward built *only* around the commander (relative pose to humanoid, following-distance/heading reward, fall/collision/lost-target termination).
3. Before spending any PPO compute, verify the plumbing with a **hand-coded P-controller** standing in for the commander (turn toward bearing, speed proportional to distance). Render a rollout. This is the checkpoint that proves the merged scene and dual-policy stepping actually work — expect this to surface subtle bugs (this session's history: obs normalization, heading-relative commands, wrapper batch-size mismatches were all found exactly this way), so budget real iteration time here rather than assuming it works first try.

**Phase 2 — Train the real commander, wrapped in the same autonomous loop.**
Once Phase 1's hand-coded controller proves the environment is correct, replace it with a PPO-trained commander (`/tmp/follow_iteration.py`, same `wrap_for_brax_training` / `make_ppo_networks` + `running_statistics.normalize` pattern already proven for Go1). Wire it into the same evaluate → detect-failure → curriculum → retrain loop used for locomotion: failure = loses target beyond a distance threshold, falls, or collides; curriculum step adjusts domain randomization (leader speed range, turn sharpness, terrain roughness) around observed failures. This should converge much faster than the 200M-step gait training since balance/gait is already solved — expect this to be the fast, iterable part, producing a basic→advanced progression (loses target constantly → keeps loose distance → tight, stable following) similar in shape to the existing locomotion progression videos.

## Files (all on the sandbox, detached-subprocess pattern, `/tmp/follow_*.py`)

- `/tmp/g1_train.py` — Phase 0, G1 humanoid training (adapted from `loco_iteration_flat.py`)
- `/tmp/follow_scene_build.py` — Phase 1, MJCF merge
- `/tmp/follow_env.py` — Phase 1, custom two-robot hierarchical `MjxEnv`
- `/tmp/follow_baseline_render.py` — Phase 1, hand-coded-controller sanity render
- `/tmp/follow_iteration.py` — Phase 2, PPO training of the commander
- `/tmp/follow_eval.py`, `/tmp/follow_curriculum.py` — Phase 2, loop plumbing (mirror `loco_eval.py` / `loco_curriculum.py`)

## Verification

- Phase 0: render the trained G1 checkpoint standalone (same technique as `loco_render_*.py`), confirm it walks stably for multiple commands before it's trusted as "frozen" for later phases.
- Phase 1: render the merged-scene rollout with the hand-coded controller; visually confirm both robots are physically present, colliding correctly with the shared floor, and the dog's torques are actually responding to the commander's output (not silently falling back to some default/frozen command — this class of "looks plausible but is actually broken" bug is exactly what burned hours on the locomotion project, so re-check observation preprocessing/normalization explicitly this time from the start rather than at the end).
- Phase 2: standard training-curve + held-out-eval check (avg episode length, average following distance) before trusting any "it learned to follow" claim, given this session's precedent that the training-time metric and a naive manual eval script can silently disagree.
