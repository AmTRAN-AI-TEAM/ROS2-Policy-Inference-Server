import rclpy
from rclpy.node import Node
import numpy as np
import os

from ament_index_python.packages import get_package_share_directory

from robomimic_policy_interfaces.srv import PolicyInfer
from robomimic_policy_srv.policy_runner import PolicyRunner

from cv_bridge import CvBridge


class PolicyServiceNode(Node):

    def __init__(self):
        super().__init__('policy_service_node')
        self.bridge = CvBridge()

        # 取得 package 安裝後的 share 目錄
        pkg_path = get_package_share_directory('robomimic_policy_srv')

        # 拼接模型路徑
        ckpt_path = os.path.join(pkg_path, 'models', 'model_epoch_20.pth')

        self.runner = PolicyRunner(ckpt_path)
        self.runner.reset()

        self.srv = self.create_service(
            PolicyInfer,
            'policy_infer',
            self.handle_request
        )

        self.get_logger().info("Policy service ready.")

    def handle_request(self, request, response):
        try:
            # 1) low_dim
            eef_pos = np.asarray(request.eef_pos, dtype=np.float32)          # (3,)
            eef_quat = np.asarray(request.eef_quat, dtype=np.float32)        # (4,)
            gripper_pos = np.asarray(request.gripper_pos, dtype=np.float32)  # (1,)

            # 2) RGB image: sensor_msgs/Image -> numpy (H,W,3) uint8
            rgb_hwc = self.bridge.imgmsg_to_cv2(request.table_cam, desired_encoding="rgb8")
            rgb_chw = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.uint8)       # (3,H,W)

            # 3) Depth image: sensor_msgs/Image -> numpy (H,W) float32
            depth_hw = self.bridge.imgmsg_to_cv2(request.table_cam_depth, desired_encoding="32FC1")
            depth_chw = depth_hw[np.newaxis, :, :].astype(np.float32)         # (1,H,W)

            # 4) 組成 robomimic 需要的 dict
            obs_dict = {
                "eef_pos": eef_pos,
                "eef_quat": eef_quat,
                "gripper_pos": gripper_pos,
                "table_cam": rgb_chw,
                "table_cam_depth": depth_chw,
            }

            # 5) 推論
            action = self.runner.step(obs_dict)  # 預期回傳 shape (7,)

            # 6) 回傳
            response.action = np.asarray(action, dtype=np.float32).tolist()
            return response

        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            # 失敗時回傳全 0 action（即時控制更安全）
            response.action = [0.0] * 7
            return response

    def convert_flat_obs(self, obs_array):
        # 這裡先寫簡單版本
        # 等下我們根據 runner.obs_shapes 自動拆分
        return {"obs": obs_array}


def main(args=None):
    rclpy.init(args=args)
    node = PolicyServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
