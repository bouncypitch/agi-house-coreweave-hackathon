# Follow-the-Humanoid: self-improving RL loop (AGI House / CoreWeave Hacks)
#
# This file is a marimo notebook. It is written and structurally checked on
# a laptop with no GPU and none of mujoco / mujoco_playground / jax[cuda]
# installed, so cells 3 onward will NOT run locally — that's expected, not
# a bug. Real execution happens on molab (see the last cell for handoff
# instructions). Every unfinished piece is marked TODO with a one-line
# contract (expected inputs/outputs) instead of fake logic.

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Follow-the-Humanoid: A Self-Improving Robot Policy Loop

        A simulated Unitree quadruped learns to follow a simulated Unitree
        humanoid in MuJoCo. Locomotion for both robots is **borrowed**
        (pretrained MuJoCo Playground joystick policies). The **built**
        piece is a small follow-controller trained with PPO in MJX, closed
        in a loop:

        `evaluate on a fixed battery -> detect failures objectively ->
        an LLM proposes new domain-randomization ranges around those
        failures -> retrain -> redeploy -> re-evaluate`

        Results log to Weights & Biases; the curriculum step is traced in
        Weave; ARIA reviews the run history near the end.
        """
    )
    return


@app.cell
def _():
    # Run this first, in molab — needs the GPU-enabled JAX build.
    # pip install mujoco mujoco-mjx mujoco_playground brax wandb weave mediapy

    import mujoco
    from mujoco import mjx
    import mujoco_playground
    from mujoco_playground import registry
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import train as ppo
    import wandb
    import weave
    import mediapy as media

    print("JAX backend:", jax.default_backend())
    return (
        jax,
        jp,
        media,
        mjx,
        mujoco,
        mujoco_playground,
        ppo,
        registry,
        wandb,
        weave,
    )


@app.cell
def _(registry):
    # TODO: confirm exact registry ids on first run in molab.
    print(dir(registry))

    QUADRUPED_ENV_ID = "Go2JoystickFlatTerrain"  # fallback: "Go1JoystickFlatTerrain"
    HUMANOID_ENV_ID = "G1JoystickFlatTerrain"  # fallback: "H1JoystickFlatTerrain"

    quadruped_env = registry.load(QUADRUPED_ENV_ID)
    humanoid_env = registry.load(HUMANOID_ENV_ID)
    return HUMANOID_ENV_ID, QUADRUPED_ENV_ID, humanoid_env, quadruped_env


@app.cell
def _(humanoid_env, jax, media, quadruped_env):
    # The single dependency the whole weekend sits on: does this render?
    # TODO: confirm this against the real Playground API on first run.
    quad_state = jax.jit(quadruped_env.reset)(jax.random.PRNGKey(0))
    human_state = jax.jit(humanoid_env.reset)(jax.random.PRNGKey(0))

    quad_frame = quadruped_env.render([quad_state.data])[0]
    human_frame = humanoid_env.render([human_state.data])[0]

    media.show_images([quad_frame, human_frame])
    return human_state, quad_state


@app.cell
def _():
    # TODO: the humanoid's moving-target path.
    # Contract: humanoid_scripted_command(t: float) -> array[3]
    #   Returns (forward_vel, lateral_vel, yaw_rate) fed into the humanoid's
    #   pretrained joystick policy at simulation time t. Should walk
    #   forward, turn, speed up, and stop over the episode.
    def humanoid_scripted_command(t):
        raise NotImplementedError("Build the waypoint/command sequence.")

    return (humanoid_scripted_command,)


@app.cell
def _():
    # TODO: the actual thing we're training this weekend.
    # Contract: make_follow_controller() -> policy
    #   policy(obs) -> array[3], where
    #   obs = concat(relative_position_to_humanoid, relative_heading,
    #                dog_proprioception)
    #   and the output is (forward_vel, lateral_vel, yaw_rate) fed into the
    #   quadruped's pretrained joystick policy.
    #
    # Reward sketch for PPO training:
    #   + maintain target follow distance/heading to the humanoid
    #   - fall / bad orientation
    #   - excessive joint effort
    #
    # Train with brax.training.agents.ppo.train, mirroring how Playground
    # trains its own joystick policies.
    def make_follow_controller():
        raise NotImplementedError("Define network + initial PPO train() call.")

    # Contract: retrain(policy, curriculum: dict) -> policy
    #   Continues PPO training from `policy`, resampling domain-
    #   randomization parameters from the ranges in `curriculum`.
    def retrain(policy, curriculum):
        raise NotImplementedError(
            "Continue PPO training with domain randomization drawn from "
            "`curriculum`'s ranges."
        )

    return make_follow_controller, retrain


@app.cell
def _():
    # TODO: objective, scripted failure criteria — the hard gate.
    # Contract: detect_failure(rollout) -> dict | None
    #   Returns a structured failure record if any condition is hit, e.g.
    #   {"type": "distance" | "fall" | "collision", "step": int,
    #    "params": {...}}, else None.
    DISTANCE_THRESHOLD_M = 3.0
    FALL_HEIGHT_THRESHOLD_M = 0.15

    def detect_failure(rollout):
        raise NotImplementedError(
            "Check distance-to-target, torso height/orientation, and "
            "collisions at each step of `rollout`."
        )

    return DISTANCE_THRESHOLD_M, FALL_HEIGHT_THRESHOLD_M, detect_failure


@app.cell
def _():
    # TODO: fixed, held-out scenarios — never used for training/augmentation.
    VALIDATION_BATTERY = [
        # {"humanoid_speed": 1.0, "turn_rate": 0.0, "obstacle": None},
        # {"humanoid_speed": 1.5, "turn_rate": 0.5, "obstacle": None},
        # add a mix of easy / hard / genuinely novel scenarios
    ]

    def evaluate(policy, battery=VALIDATION_BATTERY):
        """Run `policy` through every scenario in `battery`.

        Returns: (failure_rate: float, failure_records: list[dict])
        """
        raise NotImplementedError

    return VALIDATION_BATTERY, evaluate


@app.cell
def _():
    # TODO: one well-scoped LLM call — not a heavy agent framework.
    # Contract: propose_curriculum(failure_records: list[dict]) -> dict
    #   Input:  the batch of failure records from the last evaluate() call.
    #   Output: {"<param>": {"min": ..., "max": ...}, ...} — new domain-
    #           randomization ranges for the next training batch, biased
    #           toward the regions where failures clustered.
    def propose_curriculum(failure_records):
        raise NotImplementedError(
            "Structured LLM call: failure records in, new randomization "
            "ranges out."
        )

    return (propose_curriculum,)


@app.cell
def _(wandb):
    wandb.init(project="agi-house-follow-loop")

    def log_iteration(iteration, failure_rate, extra=None):
        payload = {"iteration": iteration, "failure_rate": failure_rate}
        if extra:
            payload.update(extra)
        wandb.log(payload)

    return (log_iteration,)


@app.cell
def _(propose_curriculum, weave):
    weave.init("agi-house-follow-loop")

    # Traced so the curriculum agent's reasoning is inspectable on stage.
    traced_propose_curriculum = weave.op()(propose_curriculum)

    return (traced_propose_curriculum,)


@app.cell
def _(evaluate, log_iteration, make_follow_controller, retrain, traced_propose_curriculum):
    # TODO: wire this up once the cells above have real implementations.
    N_ITERATIONS = 3

    def run_loop():
        policy = make_follow_controller()
        for i in range(N_ITERATIONS):
            failure_rate, failure_records = evaluate(policy)
            log_iteration(i, failure_rate)
            curriculum = traced_propose_curriculum(failure_records)
            policy = retrain(policy, curriculum)
        return policy

    return N_ITERATIONS, run_loop


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Getting this onto molab

        1. Go to molab.marimo.io and create a new notebook.
        2. Attach a GPU (RTX Pro 6000) from the notebook settings.
        3. Paste this file's contents in, or push this repo to GitHub and
           import from there.
        4. Open the actions panel (top right) -> **Pair with an agent** ->
           copy the `marimo pair` command it gives you.
        5. Run that command in a local Claude Code session to connect
           directly to the live cloud kernel — from there, edits run
           against real GPU compute instead of being guessed at locally.
        6. Run cells top to bottom. The render sanity-check cell is the
           single dependency the whole weekend sits on — confirm it works
           before building anything else.
        """
    )
    return


if __name__ == "__main__":
    app.run()
