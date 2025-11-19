"""
Train a Real RL Agent (DQN/PPO) on MCP-Gymnasium Environment

This demonstrates how to train traditional RL algorithms on your MCP environment.
We'll use Stable-Baselines3, the standard library for RL in Python.

Installation:
    pip install stable-baselines3[extra] torch tensorboard

Usage:
    python train_rl_agent.py --algorithm ppo --steps 50000
    python train_rl_agent.py --algorithm dqn --steps 100000
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from mcp import StdioServerParameters

from mcp_gym import MCPEnv
from rewards import KeywordReward, LLMReward


class DiscreteActionMCPEnv(gym.Wrapper):
    """
    Wrapper to make MCPEnv compatible with discrete action space RL algorithms.
    
    Traditional RL algorithms like DQN and PPO need:
    1. Discrete or continuous action spaces (not dict)
    2. Fixed-size observation spaces (not variable dict)
    
    This wrapper converts:
    - Actions: Dict → Integer (index into available tools + argument combinations)
    - Observations: Dict → Fixed-size vector
    """
    
    def __init__(self, env, max_tools=10, max_query_length=50):
        super().__init__(env)
        self.max_tools = max_tools
        self.max_query_length = max_query_length
        
        # Define possible search queries for demonstration
        self.search_queries = [
            "Bitcoin price",
            "Ethereum price",
            "cryptocurrency news",
            "crypto market cap",
            "latest blockchain news"
        ]
        
        # Define possible actions: 
        # - Tool 0 (web_search) with different queries
        # - Tool 1 (save_report) with fixed filename
        self.action_combinations = []
        for query in self.search_queries:
            self.action_combinations.append({
                "tool_name": "web_search",
                "arguments": {"query": query}
            })
        self.action_combinations.append({
            "tool_name": "save_report",
            "arguments": {
                "content": "Mission complete",
                "filename": "rl_report.txt"
            }
        })
        
        # Action space: discrete integer (index into action_combinations)
        self.action_space = spaces.Discrete(len(self.action_combinations))
        
        # Observation space: fixed-size vector
        # [step_count, reward_so_far, tool_0_available, tool_1_available, ...]
        obs_size = 2 + max_tools  # step count + cumulative reward + tool availability
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.available_tools = []
        
    def _encode_observation(self, obs_dict):
        """Convert dictionary observation to fixed-size vector."""
        # Extract available tools
        if "tools" in obs_dict:
            self.available_tools = [t["name"] for t in obs_dict["tools"]]
        
        # Create vector: [step_count, cumulative_reward, tool_availability...]
        vector = [self.step_count, self.cumulative_reward]
        
        # Tool availability (1 if available, 0 otherwise)
        known_tools = ["web_search", "save_report"]
        for tool in known_tools:
            vector.append(1.0 if tool in self.available_tools else 0.0)
        
        # Pad to max_tools
        while len(vector) < 2 + self.max_tools:
            vector.append(0.0)
        
        return np.array(vector[:2 + self.max_tools], dtype=np.float32)
    
    def reset(self, **kwargs):
        """Reset environment and encode observation."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.cumulative_reward = 0.0
        return self._encode_observation(obs), info
    
    def step(self, action):
        """Convert discrete action to MCP action and step."""
        # Convert integer action to MCP action dict
        mcp_action = self.action_combinations[action]
        
        # Step in underlying environment
        obs, reward, terminated, truncated, info = self.env.step(mcp_action)
        
        # Update state
        self.step_count += 1
        self.cumulative_reward += reward
        
        # Store raw action info for debugging
        info["raw_action"] = mcp_action
        info["action_index"] = action
        
        return self._encode_observation(obs), reward, terminated, truncated, info


def create_training_env(server_path, mission, algorithm="ppo"):
    """Create a training environment with proper wrappers."""
    
    # Choose reward function based on complexity
    if mission == "simple":
        reward_fn = KeywordReward(target_keywords=["Bitcoin", "Ethereum", "crypto"])
        mission_text = "Search for cryptocurrency information"
    else:
        # Load API key from .env
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        reward_fn = LLMReward(
            mission_description="Find Bitcoin and Ethereum prices, then save report",
            api_key=api_key,
            provider="groq"
        )
        mission_text = "Find Bitcoin and Ethereum prices, then save report"
    
    print(f"🎯 Mission: {mission_text}")
    print(f"🏆 Reward: {reward_fn.__class__.__name__}")
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=None
    )
    
    # Create base MCP environment
    base_env = MCPEnv(
        server_params=server_params,
        reward_function=reward_fn
    )
    
    # Wrap with discrete action space for RL algorithms
    env = DiscreteActionMCPEnv(base_env)
    
    # Monitor wrapper for tracking training metrics
    env = Monitor(env)
    
    return env


def train_agent(algorithm="ppo", total_timesteps=50000, mission="simple"):
    """Train an RL agent on the MCP environment."""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "ddg_server.py")
    
    # Create environment
    env = create_training_env(server_path, mission, algorithm)
    
    # Create vectorized environment (required by SB3)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create directories for logs and models
    log_dir = "./logs"
    model_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=f"{model_dir}/{algorithm}_best",
        log_path=f"{log_dir}/{algorithm}",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{model_dir}/{algorithm}_checkpoints",
        name_prefix=f"{algorithm}_model"
    )
    
    # Create RL agent
    print(f"\n🤖 Training {algorithm.upper()} agent...")
    print(f"📊 Total timesteps: {total_timesteps}")
    print(f"💾 Models will be saved to: {model_dir}")
    print(f"📈 Logs will be saved to: {log_dir}")
    
    if algorithm.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )
    elif algorithm.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard",
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
        )
    elif algorithm.lower() == "a2c":
        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=f"{log_dir}/tensorboard",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train the agent
    print("\n🏋️ Training started...")
    print("💡 Tip: Monitor training with: tensorboard --logdir ./logs/tensorboard")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{model_dir}/{algorithm}_final"
        model.save(final_path)
        print(f"\n✅ Training complete! Model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save(f"{model_dir}/{algorithm}_interrupted")
        print(f"💾 Model saved to: {model_dir}/{algorithm}_interrupted")
    
    finally:
        env.close()
    
    return model


def test_trained_agent(model_path, num_episodes=5):
    """Test a trained agent."""
    
    print(f"\n🧪 Testing trained agent: {model_path}")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "ddg_server.py")
    
    # Create test environment
    env = create_training_env(server_path, mission="simple", algorithm="test")
    
    # Load trained model
    algorithm = os.path.basename(model_path).split("_")[0]
    if algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    elif algorithm.lower() == "dqn":
        model = DQN.load(model_path)
    elif algorithm.lower() == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Test agent
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        print(f"\n📝 Episode {episode + 1}/{num_episodes}")
        
        while not (terminated or truncated) and step < 10:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print action taken
            if "raw_action" in info:
                print(f"  Step {step}: {info['raw_action']['tool_name']} → Reward: {reward:.2f}")
        
        episode_rewards.append(episode_reward)
        print(f"  Episode Reward: {episode_reward:.2f}")
    
    # Summary statistics
    print(f"\n📊 Test Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std Reward: {np.std(episode_rewards):.2f}")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on MCP environment")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="ppo", 
        choices=["ppo", "dqn", "a2c"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--mission",
        type=str,
        default="simple",
        choices=["simple", "complex"],
        help="Mission complexity (simple=keyword reward, complex=LLM reward)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to trained model to test (skips training)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_trained_agent(args.test)
    else:
        train_agent(args.algorithm, args.steps, args.mission)


if __name__ == "__main__":
    main()
