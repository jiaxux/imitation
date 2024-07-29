python -m imitation.scripts.eval_policy with \
        expert.policy_type=ppo \
        expert.loader_kwargs.path=/home/jixing/code/imitation/examples/playground/runs/output/ppo/halfcheetah/trained/policies/000001230000 \
        environment.num_vec=1 \
        render=True \
        environment.gym_id='HalfCheetah' \
