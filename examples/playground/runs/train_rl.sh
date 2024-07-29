 python -m imitation.scripts.train_rl with seals_ant \
        total_timesteps=1e8 \
        logging.log_dir=output/ppo/halfcheetah/trained \
        rollout_save_n_episodes=50
