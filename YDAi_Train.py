import cv2
from pywinauto import Application
import time
import random
from collections import deque
import keyboard  # 用于检测按键输入
import pytesseract
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading
import queue
from torch.utils.tensorboard import SummaryWriter


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("是否启用 GPU 加速:", torch.cuda.is_available())
print("当前 GPU 设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")

OBSERVE = 10000
EXPLORE = 5000
frame_time = 1 / 30  # 计算每帧需要的时间
class GameEnvironment:
    def __init__(self, handle, camera_index=2):
        # 连接到应用程序窗口
        self.app = Application(backend="win32").connect(handle=handle)
        self.window = self.app.window(handle=handle)
        self.recognition_queue = queue.Queue()
        self.running = False  # 初始化 running 属性
        # 初始化虚拟摄像头
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("无法打开虚拟摄像头")

        # 用于存储最近的 4 帧图像
        self.frame_stack = deque(maxlen=4)
        self.recognition_queue = queue.Queue(maxsize=1)  # 设置队列最大大小为 1

    def get_window_screenshot(self):
        """
        捕获虚拟摄像头的画面并调整大小。
        """
        ret, frame = self.cap.read()  # 捕获一帧
        if not ret or frame is None:
            print("无法捕获虚拟摄像头的画面，返回默认帧")
            return np.zeros((1674, 920, 3), dtype=np.uint8)  # 返回一个空白帧，尺寸为 1674x920

        # 调整截图尺寸为 920x1674
        resized_frame = cv2.resize(frame, (920, 1674))

        return np.array(resized_frame)  # 返回调整后的画面

    def preprocess_image(self, image):
        """
        预处理图像：调整大小、转换为灰度图像并二值化。
        """
        # 调整图像大小
        resized_image = cv2.resize(image, (80, 160))

        # 转换为灰度图像
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    def get_stacked_frames(self):
        """
        获取堆叠的图像帧。使用四帧图像来堆叠，并返回。
        """
        raw_frame = self.get_window_screenshot()
        processed_frame = self.preprocess_image(raw_frame)

        # 如果堆栈为空，用相同的帧初始化
        if len(self.frame_stack) < 4:
            for _ in range(4 - len(self.frame_stack)):
                self.frame_stack.append(processed_frame)

        # 更新帧堆栈
        self.frame_stack.append(processed_frame)

        # 只取最后 4 帧
        stacked_frames = np.array(self.frame_stack)

        #concatenated_frames = np.hstack(stacked_frames)
        #cv2.imshow("Stacked Binary Images", concatenated_frames)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return stacked_frames

    def release_camera(self):
        """
        释放摄像头资源。
        """
        if self.cap:
            self.cap.release()

    def is_game_over(self):
        # 检测“失败”界面
        screenshot = self.get_window_screenshot()
        cropped_screenshot = screenshot[1100:1150, 500:600]
        hsv_cropped = cv2.cvtColor(cropped_screenshot, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])  # 黄色下界
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_cropped, lower_yellow, upper_yellow)
        blue_green_pixels = cv2.countNonZero(mask)
        return blue_green_pixels > 50

    def perform_action(self, action):
        def click_action(coords):
            self.window.click(coords=coords)

        if action == 0:  # 向左跳
            threading.Thread(target=click_action, args=((200, 1200),)).start()
        elif action == 1:  # 向右跳
            threading.Thread(target=click_action, args=((600, 300),)).start()
        elif action == 2:
            pass
        else:
            print("wrongwrongwrong!!!!!")

    def reset(self):
        # 重置游戏
        self.window.click(coords=(780, 600))
        self.window.click(coords=(600, 1200))
        self.frame_stack.clear()  # 重置帧堆栈

    def recognize_number(self):
        # 截图数字所在区域
        x1, y1 = 380, 180  # 数字区域左上角坐标
        x2, y2 = 520, 340  # 数字区域右下角坐标
        screenshot = self.get_window_screenshot()  # 获取窗口截图
        cropped_image = screenshot[y1:y2, x1:x2]  # 截取数字区域
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Stacked Binary Images", binary_image)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()

        # 使用 pytesseract 识别数字
        recognized_text = pytesseract.image_to_string(binary_image,
                                                      config='--psm 7 -c tessedit_char_whitelist=0123456789')

        if not recognized_text.strip():
            return

        # 转换为整数奖励，默认为 0
        try:
            reward = int(recognized_text.strip())
        except ValueError:
            reward = 0  # 如果识别失败，奖励为 0
        # 将识别到的奖励放入队列中
        if self.recognition_queue.full():
            self.recognition_queue.get()  # 弹出队列中的旧值
        self.recognition_queue.put(reward)  # 放入新识别的数字

    def start_recognition_thread(self):
        # 启动数字识别线程
        if not self.running:
            self.running = True
            self.recognition_thread = threading.Thread(target=self.continuous_recognition)
            self.recognition_thread.daemon = True  # 设置为守护线程
            self.recognition_thread.start()

    def continuous_recognition(self):
        # 无限循环进行数字识别
        while self.running:
            self.recognize_number()
            time.sleep(0.03)  # 每隔1秒识别一次数字

    def stop_recognition_thread(self):
        # 停止线程
        self.running = False

    def get_recognition_reward(self):
        # 读取队列中的数字（不移除）
        if not self.recognition_queue.empty():
            return self.recognition_queue.queue[0]  # 读取队列中的第一个元素，不移除
        return 0  # 如果队列为空，返回 0

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(9216, 256)  # 输入维度是 2x2x64 = 256
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平为一维向量
        x = F.relu(self.fc1(x))
        q_values = F.relu((self.fc2(x)))
        return q_values

class DQNAgent:
    def __init__(self, action_size, gamma=0.99, epsilon=0.5, epsilon_decay=0.999999, epsilon_min=0.0001, lr=0.0001,
                 batch_size=128, memory_size=500000, n_step=4):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.batch_size = batch_size
        self.n_step = n_step  # 多步学习的步数

        self.writer = SummaryWriter(log_dir="runs/dqn_training")
        self.global_step = 0  # 初始化 global_step

        # 初始化 Q 网络和目标网络
        self.q_network = DQN(action_size).to(device)  # 移动到 device
        self.target_network = DQN(action_size).to(device)  # 移动到 device
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始化目标网络
        self.target_network.eval()  # 设置为评估模式

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        self.n_step_buffer = deque(maxlen=n_step)  # 多步缓冲区

    def choose_action(self, state, t):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 移动输入到 GPU
        if t <= OBSERVE:
            return random.randint(0, self.action_size - 1)  # 前10000步随机动作
        elif random.random() < self.epsilon:  # ε-greedy 策略
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        # 将当前步骤加入 n 步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # 如果缓冲区未满，暂不存储经验
        if len(self.n_step_buffer) < self.n_step:
            return

        # 计算 n 步回报
        n_step_reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        state, action, _, _, _ = self.n_step_buffer[0]  # 取出第一个状态和动作
        _, _, _, next_state, done = self.n_step_buffer[-1]  # 取出最后一个状态和 done 标记

        # 存储多步经验
        self.memory.append((state, action, n_step_reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 随机采样经验
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为 Tensor 并移至同一设备
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 计算当前 Q 值
        q_values = self.q_network(states).gather(1, actions)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            max_next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * max_next_q_values

        # 计算损失
        loss = F.mse_loss(q_values, target_q_values)

        # 记录损失值到 TensorBoard
        self.writer.add_scalar('Loss/train', loss.item(), self.global_step)

        # 记录 Q 值到 TensorBoard
        self.writer.add_scalar('Q_values/mean', q_values.mean().item(), self.global_step)
        self.writer.add_scalar('Q_values/max', q_values.max().item(), self.global_step)

        # 记录 epsilon 到 TensorBoard
        self.writer.add_scalar('Epsilon', self.epsilon, self.global_step)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay
        if self.global_step > EXPLORE:  # 当 global_step 超过 20000 时才开始衰减 epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # 增加全局步数
        self.global_step += 1

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())  # 更新目标网络权重

    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        print(f"模型已保存到 {filename}")

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"模型已从 {filename} 加载")


def train(env, agent, episodes=20000, target_update_freq=10):
    # 前 10000 步为观察阶段
    t = 0  # 追踪训练的总步数
    last_recognized_number = 0  # 初始化上次识别的数字

    for episode in range(episodes):
        state = env.get_stacked_frames()
        total_reward = 0
        done = False
        env.reset()

        # 检测暂停按键
        if keyboard.is_pressed("p"):
            print("训练暂停，按 'r' 恢复训练...")
            while not keyboard.is_pressed("r"):
                time.sleep(0.1)
            print("训练恢复！")

        if keyboard.is_pressed("s"):
            agent.save_model(f"dqn_model_episode_{episode + 1}.pth")

        while not done:
            start_time = time.time()
            reward = 0
            time1 = time.time()
            # 选择动作，传递当前步数 t
            action = agent.choose_action(state, t)

            env.perform_action(action)

            reward += 0.1  # 固定奖励

            # 获取识别的数字奖励
            reward_from_recognition = env.get_recognition_reward()

            # 如果数字发生变化，进行奖励更新
            if reward_from_recognition != last_recognized_number:
                #print(f"数字发生变化: 从 {last_recognized_number} 变为 {reward_from_recognition}")
                if reward_from_recognition > 0:
                    reward += 0  # 固定奖励
                last_recognized_number = reward_from_recognition

            next_state = env.get_stacked_frames()

            if env.is_game_over():
                print("over:",action)
                reward -= 1  # 游戏结束时的惩罚
                done = True
                concatenated_frames = np.hstack(next_state)
                #cv2.imshow("Stacked Binary Images", concatenated_frames)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

            # 存储经验并训练
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 时间步递增
            t += 1

            if t > OBSERVE:  # 观察期后开始训练
                agent.train()
            if t == OBSERVE:
                print("start!!!!")

            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                #print(frame_time - elapsed_time)
                time.sleep(frame_time - elapsed_time)

            #print(reward)
            #print(time.time() - time1)
        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Total Steps: {t}")


if __name__ == "__main__":

    handle = 5966368  # 替换为实际句柄
    env = GameEnvironment(handle, camera_index=2)
    env.start_recognition_thread()  # 启动数字识别线程
    action_size = 3  # 左跳、右跳、跳过
    agent = DQNAgent(action_size)
    #agent.load_model('dqn_model_episode_576.pth')
    train(env, agent)
