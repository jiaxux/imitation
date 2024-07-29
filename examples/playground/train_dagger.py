import tempfile

import numpy as np

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

env = make_vec_env(
    "HalfCheetah-v3",
    rng=np.random.default_rng(),
    n_envs=1,
)
expert = load_policy(
    "ppo",
    path="/home/jixing/code/imitation/examples/playground/runs/output/ppo/halfcheetah/trained/policies/000001430000",
    venv=env,
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=np.random.default_rng(),
    device="cuda:0",
)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(),
    )

    dagger_trainer.train(200000)
