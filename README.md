# Beam Tracker RL

Minimal Gymnasium environment for beam tracking with rectangular occluders. The agent observes only recent SNR feedback, recent beam
actions, and normalized range; privileged geometry is returned through `info`
and episode logs for debugging and plots.

## Layout

```text
beam_tracker_rl/
  sim.py      # scenario configs, geometry, channel/SNR, reward, observation helpers
  env.py      # Gymnasium Env wrapper
notebooks/
  ppo_beam_tracking.ipynb
tests/
  test_env.py
  test_sim.py
pyproject.toml
```

Removed by design: standalone debug scripts, compatibility shims, duplicated
README/spec folders, generated caches, and pinned `requirements.txt`. Install
from `pyproject.toml` instead.

## Install

```bash
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m pip install -e '.[ppo,dev]'
```

## Use

```python
import gymnasium as gym
import beam_tracker_rl
from beam_tracker_rl import ENV_ID

env = gym.make(ENV_ID, scenario_name="single_occluder")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Tune scenarios directly through environment kwargs:

```python
from beam_tracker_rl import (
    BeamTrackingEnv,
    ChannelConfig,
    MovementConfig,
    Obstacle,
    RewardConfig,
)

env = BeamTrackingEnv(
    scenario_name="single_occluder",
    obstacles=(Obstacle(x=470.0, y=220.0, w=84.0, h=140.0),),
    num_beams=17,
    channel_config=ChannelConfig(blockage_loss_db=18.0),
    reward_config=RewardConfig(switch_weight=0.01),
    movement_config=MovementConfig(model="stochastic"),
)
```

Movement is deterministic constant velocity by default. Use
`MovementConfig(model="stochastic", ...)` to add seeded random speed and heading
variation with reflected bounds.

Record visualization data by passing `data_dir`. Each environment instance
creates a timestamped run folder containing per-episode `metadata.json` and
`steps.csv` files with UE position, selected and optimal beams, SNR, outage,
reward terms, and event flags.

```python
env = BeamTrackingEnv(data_dir="recordings/manual_run", run_name="eval")
```

Open [notebooks/ppo_beam_tracking.ipynb](notebooks/ppo_beam_tracking.ipynb) for
parameter tuning and PPO training/evaluation.
Open [notebooks/ppo_from_scratch_beam_tracking.ipynb](notebooks/ppo_from_scratch_beam_tracking.ipynb)
for a PPO implementation written directly with PyTorch.

## Test

```bash
./.venv/bin/python -m pytest
```
