import tempfile

import gymnasium as gym

import numpy as np

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

env = make_vec_env(
    "HalfCheetah-v4",
    rng=np.random.default_rng(),
    n_envs=5,
    env_make_kwargs={"render_mode": "rgb_array"},
)
expert = load_policy(
    "ppo",
    path="output/ppo/halfcheetah/trained/policies/final/model",
    venv=env,
)

image_observation_space = gym.spaces.Box(
    low=0,
    high=255,
    shape=(480, 480, 3),
    dtype=np.uint8,
)

bc_trainer = bc.BC(
    observation_space=image_observation_space,
    action_space=env.action_space,
    rng=np.random.default_rng(),
    device="cuda:0",
)

dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir="output/dagger/halfcheetah",
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=np.random.default_rng(),
)

dagger_trainer.train(10)
bc_trainer.train(n_epochs=10)
dagger_trainer.save_trainer()
