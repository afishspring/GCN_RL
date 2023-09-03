import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from RumorControl import rumor  # 导入您的rumor环境类

# 检查是否有可用的GPU，如果有则使用第一个可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建自定义环境
env = rumor()
env = DummyVecEnv([lambda: env])

# 初始化DQN模型并将其移动到GPU上
model = DQN(
    "MlpPolicy",
    env=env,
    device="cuda",
    verbose=1,
    batch_size=64,
    learning_rate=0.001,
    learning_starts=1000,
    target_update_interval=500,
)

checkpoint_interval = 15625
checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path="trained_models", name_prefix="ppo_rumor")

# 训练模型
total_timesteps = int(1e5)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback], log_interval=1000)

# 保存训练好的模型
model.save("rumor_dqn_model")

# 加载已训练的模型并将其移动到GPU上
loaded_model = DQN.load("rumor_dqn_model")

# 评估模型
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")