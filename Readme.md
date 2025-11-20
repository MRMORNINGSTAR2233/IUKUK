# IUKUK - AI Research Projects

This repository contains experimental AI and machine learning research projects.
python train_rl_agent.py --algorithm ppo --steps 50000 --mission simple

# Monitor training
tensorboard --logdir ./logs/tensorboard

# Test trained model
python train_rl_agent.py --test ./models/ppo_best/best_model.zip