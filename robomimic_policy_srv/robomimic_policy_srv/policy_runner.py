import torch
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


class PolicyRunner:

    def __init__(self, ckpt_path, device=None):

        if device is None:
            device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        self.device = device

        self.rollout_policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device
        )

        self.algo = self.rollout_policy.policy

        self.obs_shapes = self.algo.obs_shapes
        self.action_dim = self.algo.ac_dim

        print("Policy loaded.")
        print("Required inputs:", list(self.obs_shapes.keys()))
        print("Action dim:", self.action_dim)

    def reset(self):
        self.rollout_policy.start_episode()

    def _preprocess_obs(self, obs_dict):

        processed = {}

        for key, shape in self.obs_shapes.items():

            data = obs_dict[key]
            tensor = torch.from_numpy(data).float()

            # 只加 batch
            if tensor.ndim == len(shape):
                tensor = tensor.unsqueeze(0)

            processed[key] = tensor.to(self.device)

        return processed

    @torch.no_grad()
    def step(self, obs_dict):

        processed_obs = self._preprocess_obs(obs_dict)

        # API 測試模式：直接用 policy.get_action
        action = self.rollout_policy.policy.get_action(
            obs_dict=processed_obs,
            goal_dict=None
        )

        # RNNGMM 會回傳 (B, T, action_dim)
        if isinstance(action, tuple):
            action = action[0]

        action = action.squeeze(0).squeeze(0)

        return action.cpu().numpy()

# =========================================================
# 測試入口
# =========================================================

if __name__ == "__main__":

    ckpt_path = "/home/robertlo/IsaacLab/logs/robomimic_perry_test_0226_1/Isaac-TM5S/bc_rnn_image_franka_stack/20260226145149/models/model_epoch_20.pth"

    runner = PolicyRunner(ckpt_path)
    runner.reset()

    fake_obs = {}

    for key, shape in runner.obs_shapes.items():
        fake_obs[key] = np.random.randn(*shape).astype(np.float32)

    print("\n===== Fake OBS =====")
    for k, v in fake_obs.items():
        print(f"\n{k}:")
        print("shape:", v.shape)
        print(v)

    action = runner.step(fake_obs)

    print("\n===== Output Action =====")
    print("Action shape:", action.shape)
    print("Action value:", action)

# =========================================================
# 正式 API 介面（供外部程式 import 使用）
# =========================================================

def load_policy(ckpt_path, device=None):
    """
    載入模型並初始化 RNN 狀態
    """
    runner = PolicyRunner(ckpt_path, device=device)
    runner.reset()
    return runner


def predict(runner, obs_dict):
    """
    輸入 obs_dict，回傳 action (7,)
    """
    return runner.step(obs_dict)

# 未來使用
# from policy_runner import load_policy, predict
# ckpt = "model.pth"
# runner = load_policy(ckpt)
# action = predict(runner, obs_dict)