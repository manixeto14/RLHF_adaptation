# RLHF in Industrial Adaptive User Interfaces

This repository contains a research project exploring the use of Reinforcement Learning from Human Feedback (RLHF) to optimize and automatically adapt industrial User Interfaces (UIs) based on operator context (time, fatigue, shift, familiarity).

## Project Workflow

The project uses a dataset of recorded user interactions. To simulate these real interactions, we encapsulate the dataset within a Gymnasium (`gym`) environment. 

1. **State / Observation:** The environment observes the context of the user (e.g., Time of the day, User ID, Shift type, Day of the week).
2. **Action / Adaptation:** The RL agent generates an interface adaptation. It defines the layout by choosing the `mixture` (widgets/components) shown on screen in 16 available slots, selecting the `additive` level, and toggling a `container` status.
3. **Reward:** After generating the UI adaptation, the system evaluates it by comparing the agent's proposed interface against the historically recorded preference of that user at that specific time. The reward function essentially simulates the mathematical "satisfaction" of the user.
4. **Optimization:** The RL policy is updated to maximize this long-term inferred satisfaction.

---

## Architectural Approaches (Gym Environments)

We have two distinct mathematical approaches to modeling this problem, which require different algorithm considerations:

### 1. The Contextual Bandit Approach (`src/env/env.py`)
- **Concept:** The agent provides the *full interface configuration* in a single action step.
- **Action Space:** `MultiDiscrete`. It chooses all 16 slots, the additive, and the container simultaneously. 
- **Algorithms:** Requires algorithms capable of massive combinatorial output spaces (like PPO). Standard DQN cannot be applied here because the state-action space is too big ($17^{16} \times 6 \times 2 \approx 5.8 \times 10^{20}$).
- **Training Script:** `train.py`

### 2. The Sequential Builder Approach (`src/env/builder_env.py`)
- **Concept:** The agent acts as an *Architect*, building the interface step-by-step iteratively over 18 steps per episode.
- **Action Space:** `Discrete(17)`. The agent only takes one small localized decision at a time (e.g., "What goes in slot 1?", "What is the additive level?").
- **State Space Upgrade:** The state now includes a `canvas` array tracking the construction progress, and biological simulation markers like `fatigue_level`.
- **Reward Dynamics:** Known as a *Sparse Reward* problem. The agent receives 0 reward while building. Only at step 18, when the UI is complete, does the environment calculate satisfaction and deliver the reward payload. 
- **Algorithms:** Because the action space is a simple `Discrete`, this environment is 100% compatible with traditional Deep Q-Learning (DQN) arrays, while still supporting PPO.
- **Training Scripts:** `train_builder_dqn.py` and `train_builder_ppo.py`.

---

## Running the Code

Ensure you have your dataset placed at `data/cooked_train.csv`.

To test the traditional PPO (All at once):
```bash
python train.py
```

To compare DQN vs PPO using the Sequential Builder methodology:
```bash
python train_builder_dqn.py
python train_builder_ppo.py
```
Results and models will be stored in the `/results` directory.
