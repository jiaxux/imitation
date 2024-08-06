python -m imitation.scripts.eval_policy with \
        expert.policy_type=ppo \
        expert.loader_kwargs.path=/home/jixing/Code/imitation/examples/playground/runs/output/ppo/halfcheetah/trained/policies/final \
        environment.num_vec=1 \
        environment.gym_id='HalfCheetah-v4' \
