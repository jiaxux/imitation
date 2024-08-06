MUJOCO_GL=egl python ../train_dagger.py with half_cheetah \
       total_timesteps=1e8 \
       logging.log_dir=output/dagger/halfcheetah/trained \
       rollout_save_n_episodes=50 
