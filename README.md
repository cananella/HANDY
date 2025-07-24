# HANDY

**HANDY** (Humanoid Agent for Navigating and Doing for You) is an open-source project that develops a humanoid robot assistant capable of navigating realistic environments and performing household tasks — intelligently, autonomously, and in simulation.

Built on top of [ManiSkill3](https://github.com/haosulab/ManiSkill), HANDY provides simulation-ready environments, modular agents, and embodied task implementations to support research in home robotics, cognitive control, and embodied AI.


## Project Goals

- Explore how embodied humanoid agents can learn to assist in household contexts
- Integrate reactive behaviors and deliberative planning in simulation
- Build reusable components for perception, reasoning, and action
- Support generalization across home-like environments and tasks

## Features

- 🧍‍♂️ Humanoid robot simulation using ManiSkill3
- 🏠 Realistic domestic environments for household assistance tasks
- 🧠 Dual-mode cognitive architecture (System 1 & System 2 integration)
- 🛠️ Modular design & open-source contributions welcome

## Installation

We use [`uv`](https://github.com/astral-sh/uv), a modern Python package manager, for fast, reproducible development environments.

### Step-by-step

```bash
# 1. Install uv (if not already)
curl -Ls https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/AuTURBO/HANDY.git
cd HANDY

# 3. Add dependencies
uv add torch
uv add --editable ./third_party/ManiSkil
```

## Project Structure (Top Level)

- `projects/` — Example experiments and task-specific scripts
- `src/` — Source code for environments, robots, and simulation logic
- `third_party/` — External dependencies (e.g., ManiSkill, robot models)
- `pyproject.toml` — Project configuration for Python and uv
- `uv.lock` — Reproducible dependency lockfile (used by uv)

# Supported Robots
HANDY supports various robots available in ManiSkill3,
but we primarily use the **[ROBOTIS AI Worker](https://ai.robotis.com/)** as the main robot platform for our simulations and tasks.
The AI Worker is a semi-humanoid (25 DOF) robot optimized for bimanual manipulation, ROS 2 integration, and Physical AI workflows—offering seamless simulation, teleoperation, and autonomous policy execution.

The AI Worker model is integrated into our project and serves as the default agent for embodied task execution.


## Quick Start

> uv run python projects/spatial_memory/robot_test.py -r "ai_worker"


## Citation

Coming soon.

