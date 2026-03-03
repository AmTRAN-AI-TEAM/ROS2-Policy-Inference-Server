# ROS2 Policy Inference Server

本專案提供一個基於 ROS2 的推論服務（Inference Server），可將 robomimic 訓練完成的 policy 模型封裝為 ROS2 Service。

使用者只需提供格式正確的 observation（包含 low_dim + RGB + depth），即可獲得對應的 7 維 action 輸出。

---

## 一、系統架構
```
外部控制端 (Client)
        ↓
/policy_infer (ROS2 Service)
        ↓
policy_service_node.py
        ↓
PolicyRunner
        ↓
robomimic 模型 (.pth)
        ↓
回傳 action (7D)
```

> 本專案僅負責「推論」，不包含訓練流程。

---

## 二、環境需求

| 項目 | 版本 |
|------|------|
| 作業系統 | Ubuntu 22.04 |
| ROS2 | Humble |
| Python | 3.10 |
| robomimic | 最新穩定版 |
| torch | 最新穩定版 |
| torchvision | 最新穩定版 |
| cv_bridge | — |
| numpy | `< 2.0` |

> ⚠️ 建議將 ML 套件安裝於獨立 virtualenv（例如 `ros2_ml_env`）

---

## 三、安裝方式

### 1️⃣ 下載專案
```bash
cd ~/ros2_ws/src
git clone https://github.com/AmTRAN-AI-TEAM/ROS2-Policy-Inference-Server.git
```

### 2️⃣ 編譯
```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 四、模型權重設定

> ⚠️ 本專案不包含模型權重檔，請自行準備已訓練完成的 robomimic 模型。

請將模型檔放置於以下路徑：
```
robomimic_policy_srv/models/model_epoch_20.pth
```

或自行修改 `policy_service_node.py` 內的模型路徑設定。

---

## 五、啟動推論服務

請使用已安裝 robomimic / torch 的虛擬環境：
```bash
source ~/ros2_ml_env/bin/activate
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python -m robomimic_policy_srv.policy_service_node
```

成功啟動後應看到：
```
Policy loaded.
Policy service ready.
```

---

## 六、Service 介面說明

**Service 名稱：** `/policy_infer`

### 輸入（Request）

| 欄位名稱 | 型別 | 說明 |
|----------|------|------|
| `eef_pos` | `float32[3]` | 末端位置 |
| `eef_quat` | `float32[4]` | 末端四元數 |
| `gripper_pos` | `float32[1]` | 夾爪位置 |
| `table_cam` | `sensor_msgs/Image (RGB8)` | RGB 影像 |
| `table_cam_depth` | `sensor_msgs/Image (32FC1)` | 深度影像 |

### 輸出（Response）

| 欄位名稱 | 型別 | 說明 |
|----------|------|------|
| `action` | `float32[7]` | Policy 輸出的 7 維控制向量 |

---

## 七、預期使用流程

1. 收集 observation（位置、姿態、影像等）
2. 呼叫 `/policy_infer` Service
3. 取得 7 維 action 輸出
4. 將 action 傳送至機械手臂控制器

---

## 八、注意事項

- 本專案**僅負責推論**，不包含模型訓練流程
- 模型結構需與訓練時的 observation spec **完全相符**
- 建議加入 shape 驗證機制以提高推論穩定性
- 若出現 numpy 版本錯誤，請確認已安裝 `numpy < 2.0`
