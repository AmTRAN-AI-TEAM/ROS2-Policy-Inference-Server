import rclpy
from rclpy.node import Node
import numpy as np

from robomimic_policy_interfaces.srv import PolicyInfer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PolicyClient(Node):

    def __init__(self):
        super().__init__('policy_test_client')
        self.cli = self.create_client(PolicyInfer, 'policy_infer')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.req = PolicyInfer.Request()
        self.bridge = CvBridge()

    def send_request(self):

        # -------------------------
        # 1️⃣ low_dim fake data
        # -------------------------
        self.req.eef_pos = np.random.randn(3).astype(np.float32).tolist()
        self.req.eef_quat = np.random.randn(4).astype(np.float32).tolist()
        self.req.gripper_pos = np.random.randn(1).astype(np.float32).tolist()

        # -------------------------
        # 2️⃣ Fake RGB image
        # robomimic 要 (3, H, W)
        # 但 ROS message 是 (H, W, 3)
        # -------------------------
        H, W = 480, 640
        rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        self.req.table_cam = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")

        # -------------------------
        # 3️⃣ Fake depth
        # -------------------------
        depth = np.random.randn(H, W).astype(np.float32)
        self.req.table_cam_depth = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")

        return self.cli.call_async(self.req)


def main():
    rclpy.init()
    client = PolicyClient()

    future = client.send_request()
    rclpy.spin_until_future_complete(client, future)

    if future.result() is not None:
        print("\n====== ACTION RECEIVED ======")
        print(future.result().action)
    else:
        print("Service call failed")

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
