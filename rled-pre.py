import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
from sklearn.ensemble import RandomForestClassifier

# 假设已经从error_detection.py导入错误检测方法
from error_detection import nadeef_error_detection, dboost_error_detection, outlier_error_detection, mv_error_detection\
    , RandomForest_error_detection,FAHES_error_detection, KATARA_error_detection,ZeroER_error_detection


class ErrorDetectionEnv:
    def __init__(self, dirty_data, clean_data,methods):
        """
        初始化环境，接收脏数据、干净数据和错误检测方法列表
        """
        self.dirty_data_path = dirty_data  # 这里直接保存路径
        self.clean_data_path = clean_data  # 这里直接保存路径
        self.methods = methods
        self.dirty_data = pd.read_csv(dirty_data)  # 加载脏数据
        #self.dirty_data = self.dirty_data.drop(columns=['tuple_id'], errors='ignore')
        self.clean_data = pd.read_csv(clean_data)  # 加载干净数据
        self.n_features = self.dirty_data.drop(columns=['weight', 'fields_to_replace'], errors='ignore').shape[1]
        self.state_size = self.get_state().shape[0]  # 取 get_state() 返回的特征数
        self.current_step = 0
        self.done = False
        self.max_steps = 1

        # # 获取第一列的列名
        # self.id_column = self.dirty_data.columns[0]
        #
        # # 计算特征数时排除 id列、weight列 和 fields_to_replace列
        # self.n_features = self.dirty_data.drop(
        #     columns=[self.id_column, 'weight', 'fields_to_replace'],
        #     errors='ignore'
        # ).shape[1]

    def reset(self):
        """
        重置环境并返回初始状态
        """
        self.current_step = 0
        self.dirty_data = pd.read_csv(self.dirty_data_path)
        state = self.get_state()
        return state

    def get_state(self):
        """
        获取当前状态：返回每个特征的缺失值比例
        """
        missing_ratio = self.dirty_data.drop(columns=['weight', 'fields_to_replace'], errors='ignore').isnull().mean().values
        return missing_ratio.astype(np.float32)

    def step(self, actions):
        """
        执行选择的错误检测方法
        """
        self.current_step += 1
        applied_methods = []
        rewards = []

        # 获取所有被选择的方法
        selected_methods = [agent_id for agent_id, action in enumerate(actions) if action > 0.5]

        # 如果没有被选择的方法，直接返回
        if not selected_methods:
            next_state = self.get_state()
            reward = self.calculate_reward(applied_methods)
            rewards = [reward for _ in actions]
            return next_state, rewards, self.done, applied_methods

        temp_dirty_data = self.dirty_data.copy()  # 复制原始数据

        #print(actions)
        for agent_id in selected_methods:
            # 确保 action 是整数
            # if isinstance(action, np.ndarray):  # 如果是 NumPy 数组
            #     action = int(action.item())  # 提取单个值
            # else:
            #     action = int(action)  # 如果已经是标量，则直接转换
            # method = self.methods[action]

            if agent_id == 0:
                detected_data = outlier_error_detection(self.clean_data_path, self.dirty_data_path)
            elif agent_id == 1:
                detected_data = mv_error_detection(self.dirty_data_path)
            elif agent_id == 2:
                detected_data = nadeef_error_detection(self.clean_data_path, self.dirty_data_path,
                                                       dataset_name='beer')
            elif agent_id == 3:
                detected_data = dboost_error_detection(self.clean_data_path, self.dirty_data_path)
            # elif agent_id == 3:
            #     detected_data = RandomForest_error_detection(self.dirty_data_path, self.clean_data_path)
            elif agent_id == 4:
                detected_data = FAHES_error_detection(self.dirty_data_path)
            elif agent_id == 5:
                detected_data = KATARA_error_detection(self.clean_data_path, self.dirty_data_path,
                                                       knowledge_base_path='global_knowledge_base.json',
                                                       dataset_name='beer')
            elif agent_id == 6:
                detected_data = ZeroER_error_detection(self.dirty_data_path)
            else:
                return

            # 检查 detected_data 是否为 None
            if detected_data is None:
                print(f"Warning: Error detection method {agent_id} returned None!")
                continue  # 跳过这个方法的处理

            applied_methods.append(agent_id)
            # **合并错误检测结果**
            temp_dirty_data = self.merge_error_detection_results(temp_dirty_data, detected_data)

        self.dirty_data = temp_dirty_data.copy()  # 更新脏数据
        next_state = self.get_state()
        reward = self.calculate_reward(applied_methods)
        rewards = [reward for _ in actions]
        self.done = False

        return next_state, rewards, self.done, applied_methods

    def merge_error_detection_results(self, base_data, new_data):
        """
        合并错误检测结果：
        1. weight 列：如果任意一个方法检测出错误，则设为 1
        2. fields_to_replace 列：合并不同方法检测出的错误字段
        """
        if 'weight' not in base_data.columns:
            base_data['weight'] = 0
        if 'fields_to_replace' not in base_data.columns:
            base_data['fields_to_replace'] = [[] for _ in range(len(base_data))]

        # 确保 new_data 包含需要的列
        if 'weight' in new_data.columns:
            base_data['weight'] = base_data['weight'] | new_data['weight']  # 任何方法检测出错误，都设为1

        if 'fields_to_replace' in new_data.columns:
            # 合并检测到的字段，不同方法检测到的错误字段会合并为一个列表
            for idx in range(len(base_data)):
                # 获取已经检测出的错误字段
                existing_fields = set(base_data['fields_to_replace'][idx])
                # 获取当前方法检测到的字段
                new_fields = set(new_data['fields_to_replace'][idx])
                # 合并字段
                base_data.at[idx, 'fields_to_replace'] = list(existing_fields.union(new_fields))

        return base_data

    # def calculate_reward(self,applied_methods):
    #     """
    #     计算奖励：基于检测到的错误单元格的数量
    #     """
    #     # 计算检测到的错误单元格数量
    #     # new_errors = self.dirty_data['weight'].sum()  # 'weight'列为1时表示检测到错误
    #     # detection_reward = new_errors * 0.5  # 给予每个新检测到的错误一个奖励值
    #
    #     correct_detection = 0
    #     false_detection = 0  # 错误检测到的错误
    #     missed_detection = 0  # 实际错误未检测到的错误
    #     total_detected = 0
    #
    #     # 遍历每行数据，检查每个错误字段
    #     for index, row in self.dirty_data.iterrows():
    #         detected_fields = row['fields_to_replace']  # 获取此行检测到的错误字段
    #
    #         # 如果检测到错误，遍历所有错误字段
    #         if detected_fields != []:
    #             total_detected += len(detected_fields)
    #
    #             # 检查每个字段是否为实际错误
    #             for field in detected_fields:
    #                 if row[field] != self.clean_data.at[index, field]:  # 如果脏数据与干净数据不同，则为错误
    #                     correct_detection += 1
    #                 else:
    #                     false_detection += 1  # 误报：错误字段被误认为错误
    #         else:
    #             # 如果没有检测到错误，检查是否有实际错误
    #             for field in self.clean_data.columns:
    #                 if pd.isnull(self.dirty_data.at[index, field]) and not pd.isnull(self.clean_data.at[index, field]):
    #                     missed_detection += 1  # 漏检：实际错误未被检测到
    #
    #     # 计算准确率：正确检测的错误 / 总检测到的错误
    #     precision = correct_detection / (correct_detection + false_detection) if (correct_detection + false_detection) > 0 else 0
    #     recall = correct_detection / (correct_detection + missed_detection) if (correct_detection + missed_detection) > 0 else 0
    #     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #
    #     #计算方法多样性奖励（如果选择的方法与上一次不同，则增加奖励）
    #     diversity_reward = 0
    #     if hasattr(self, "last_applied_methods"):
    #         unique_new_methods = set(applied_methods) - set(self.last_applied_methods)
    #         diversity_reward = len(unique_new_methods) * 0.05  # 每个新方法增加0.05奖励
    #
    #     self.last_applied_methods = applied_methods  # 记录本轮选择的方法
    #
    #     # 奖励可以根据错误检测的数量来设定，检测到更多错误就给予更高奖励
    #     reward = (f1_score + diversity_reward) * 100
    #     #reward = f1_score * 100
    #     return reward

    def calculate_reward(self, applied_methods):
        """
        计算奖励：基于每个单元格的投票数量和错误概率估计
        """
        total_votes = np.zeros(self.dirty_data.shape[0])  # 初始化每个单元格的投票数量
        total_errors = 0  # 统计总的错误数量

        # 遍历每个检测器
        for agent_id in applied_methods:
            detected_data = self.get_detected_data(agent_id)  # 获取当前检测器的检测结果

            # 检查 detected_data 是否为空
            if detected_data is None or detected_data.empty:
                print(f"Warning: Detected data for agent {agent_id} is empty or None.")
                continue  # 跳过当前检测器

            # 遍历每个单元格
            for index in range(len(self.dirty_data)):
                if index in detected_data.index and detected_data.loc[index].any():  # 使用 loc 按标签访问
                    total_votes[index] += 1  # 增加该单元格的投票数量

        # 计算总的错误数量
        total_errors = np.sum(total_votes)

        # 计算错误概率估计
        error_probability_estimate = total_errors / len(self.dirty_data) if len(self.dirty_data) > 0 else 0

        # 计算投票方差
        vote_variance = np.var(total_votes)  # 计算投票结果的方差

        # 计算方法多样性奖励（如果选择的方法与上一次不同，则增加奖励）
        diversity_reward = 0
        if hasattr(self, "last_applied_methods"):
            unique_new_methods = set(applied_methods) - set(self.last_applied_methods)
            diversity_reward = len(unique_new_methods) * 0.1  # 每个新方法增加0.1奖励

        self.last_applied_methods = applied_methods  # 记录本轮选择的方法

        # 计算动态奖励函数
        reward = (1 - vote_variance) * 0.8 + diversity_reward * 0.2
        #reward = (1 - vote_variance) * 0.8

        # 确保奖励不低于0
        final_reward = max(reward, 0)

        return final_reward

    def get_detected_data(self, agent_id):
        """
        获取当前检测器的检测结果
        """
        if agent_id == 0:
            return outlier_error_detection(self.clean_data_path, self.dirty_data_path)
        elif agent_id == 1:
            return mv_error_detection(self.dirty_data_path)
        elif agent_id == 2:
            return nadeef_error_detection(self.clean_data_path, self.dirty_data_path,
                                          dataset_name='beer')
        elif agent_id == 3:
            return dboost_error_detection(self.clean_data_path, self.dirty_data_path)
        # elif agent_id == 3:
        #     return RandomForest_error_detection(self.dirty_data_path, self.clean_data_path)
        elif agent_id == 4:
            return FAHES_error_detection(self.dirty_data_path)
        elif agent_id == 5:
            return KATARA_error_detection(self.clean_data_path, self.dirty_data_path,
                                          knowledge_base_path='global_knowledge_base.json',
                                          dataset_name='beer')
        elif agent_id == 6:
            return ZeroER_error_detection(self.dirty_data_path)
        else:
            return None



# 强化学习智能体类
class Agent:
    def __init__(self, state_size, action_size, agent_id, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.num_agents = num_agents

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.update_target_network(1.0)

    def build_actor(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Sigmoid()
        )
        return model

    def build_critic(self):
        model = nn.Sequential(
            nn.Linear(self.state_size * self.num_agents + self.action_size * self.num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        return model

    def act(self, state,noise_scale):
        # if random.random() < epsilon:0.1
        #     return np.random.randint(0, self.action_size)  # 以epsilon概率随机选择方法
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        state = torch.FloatTensor(state)
        action = self.actor(state)
        action = action.detach().numpy()
        noise = noise_scale * np.random.randn(*action.shape)
        action = np.clip(action + noise, 0.0, 1.0)
        return action

    def target_act(self, state):
        state = torch.FloatTensor(state)
        action = self.target_actor(state)
        return action.detach().numpy()

    def update_target_network(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# 定义多代理强化学习环境
class MADDPG:
    def __init__(self, num_agents, state_size, action_size):
        self.agents = [Agent(state_size, action_size, agent_id, num_agents) for agent_id in range(num_agents)]
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(1000000)

        self.batch_size = 1024
        self.gamma = 0.95
        self.tau = 0.01

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放池中采样
        samples = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = samples

        # 将数据转换为张量
        states_tensor = torch.FloatTensor(states.reshape(self.batch_size, -1))
        actions_tensor = torch.FloatTensor(actions.reshape(self.batch_size, -1))
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states.reshape(self.batch_size, -1))

        for agent in self.agents:
            # 更新 Critic 网络
            # 下一个状态的所有动作
            next_actions = [self.agents[i].target_act(next_states[:, i, :]) for i in range(self.num_agents)]
            next_actions_tensor = torch.FloatTensor(np.concatenate(next_actions, axis=1))

            # 计算目标Q值
            target_inputs = torch.cat([next_states_tensor, next_actions_tensor], dim=1)
            with torch.no_grad():
                target_Q = rewards_tensor[:, agent.agent_id].unsqueeze(1) + self.gamma * agent.target_critic(
                    target_inputs)

            # 当前Q值
            current_inputs = torch.cat([states_tensor, actions_tensor], dim=1)
            current_Q = agent.critic(current_inputs)

            # 计算 Critic 损失
            critic_loss = nn.MSELoss()(current_Q, target_Q)

            # 更新 Critic 网络
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # 更新 Actor 网络
            # 当前智能体的状态
            agent_state = torch.FloatTensor(states[:, agent.agent_id, :])
            agent_action = agent.actor(agent_state)

            # 其他智能体的动作保持不变
            other_actions = []
            for i in range(self.num_agents):
                if i == agent.agent_id:
                    other_actions.append(agent_action)
                else:
                    other_actions.append(torch.FloatTensor(actions[:, i, :]))

            all_actions = torch.cat(other_actions, dim=1)
            actor_inputs = torch.cat([states_tensor, all_actions], dim=1)

            # 计算 Actor 损失
            actor_loss = -agent.critic(actor_inputs).mean()

            # 更新 Actor 网络
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # 更新目标网络
            agent.update_target_network(self.tau)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states):
        self.buffer.append((states, actions, rewards, next_states))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


# 主程序
def main():
    # 加载数据
    # dirty_data = pd.read_csv('nasa_60%.csv')
    # clean_data = pd.read_csv('nasa_clean.csv')
    dirty_data_path = 'beers_dirty.csv'
    clean_data_path = 'beers_clean.csv'
    # dirty_data_path = 'nasa_80%.csv'
    # clean_data_path = 'nasa_clean.csv'

    methods = ['outlier', 'mv', 'NADEEF','dBoost','FAHES','KATARA','ZeroER']
    #methods = ['outlier', 'mv', 'dBoost', 'KATARA','ZeroER']
    #methods = ['outlier', 'mv','NADEEF', 'dBoost', 'KATARA', 'ZeroER']
    #methods = ['outlier', 'mv', 'NADEEF', 'dBoost',  'ZeroER']
    #methods = ['mv','NADEEF','dBoost', 'KATARA','ZeroER']

    env = ErrorDetectionEnv(dirty_data_path, clean_data_path, methods)

    num_agents = len(methods)
    state_size = env.dirty_data.shape[1]
    action_size = 1
    # print(f"State size (expected by model): {state_size}")
    # print(f"State shape at reset: {env.reset().shape}")

    maddpg = MADDPG(num_agents, state_size, action_size)

    # 训练过程
    num_episodes = 100
    max_steps = env.max_steps

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}: Starting training...")  # 输出当前训练周期提示
        state = env.reset()

        #epsilon = max(0.1, 0.99 ** episode)  # 逐步减少随机探索
        # 在每个 episode 内，只选择一次方法进行错误检测
        applied_methods = []
        # **选择方法**（确保不重复选择相同的方法）
        for step in range(max_steps):
            selected_actions = set()
            actions = []
            noise_scale = max(0.1, 0.99 ** episode)
            for agent in maddpg.agents:
                action = agent.act(state.reshape(1, -1), noise_scale=noise_scale)
                #action = int(agent.act(state.reshape(1, -1), noise_scale))
                # if isinstance(agent.act(state.reshape(1, -1), noise_scale), np.ndarray):  # 如果是 NumPy 数组
                #     action = int(agent.act(state.reshape(1, -1), noise_scale).item())  # 提取单个值
                # else:
                #     action = int(agent.act(state.reshape(1, -1), noise_scale))  # 如果已经是标量，则直接转换
                #print(action)
                actions.append(action)
                # if action not in selected_actions:  # 确保不重复
                #     selected_actions.add(action)
                #     actions.append(action)

            next_state, rewards, done, applied_methods = env.step(actions)
            # method_names = [methods[method_id] for method_id in applied_methods]
            method_names = [methods for methods in applied_methods]  # 直接使用方法名称列表
            print(f"Episode {episode + 1}, Methods applied: {method_names}")

            # 存储和更新
            maddpg.memory.push(state, actions, rewards, next_state)
            maddpg.update()

            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes} finished")

        #保存每一回合的dirty_data
        # output_filename = f"dirty_data_episode_{episode + 1}.csv"
        # env.dirty_data.to_csv(output_filename, index=False)
        # print(f"Episode {episode + 1}/{num_episodes} finished, saved to {output_filename}")

    # 在训练结束后进行迭代修复过程
    max_iterations = 10  # 迭代次数
    best_reward = float('-inf')  # 初始化奖励为负无穷
    best_iteration_data = None  # 用来存储最佳奖励的迭代数据
    best_iteration = 0  # 记录奖励最高的迭代次数

    # 重置脏数据
    state = env.get_state()

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")
        # 每次迭代开始时，重置脏数据
        env.reset()
        state = env.get_state()

        # 在迭代过程中选择错误检测方法
        selected_actions = set()
        actions = []
        applied_methods = []
        for agent in maddpg.agents:
            action = agent.act(state.reshape(1, -1), noise_scale=noise_scale)
            # if action not in selected_actions:  # 确保不重复
            #     selected_actions.add(action)
            #     actions.append(action)
            actions.append(action)

        # 执行错误检测方法
        next_state, rewards, done, applied_methods = env.step(actions)

        # 输出本次迭代的结果
        method_names = [methods for methods in applied_methods]
        print(f"Iteration {iteration + 1}: Selected Detect Methods: {method_names}, Reward(f1_score + diversity_reward): {rewards[0]}")
        # print(f"Selected Detect Methods: {method_names}")
        # print(f"Reward: {rewards[0]}")

        # 检查当前奖励并保存奖励最高的迭代
        current_reward = rewards[0]  # 获取当前迭代的奖励
        if current_reward > best_reward:
            best_reward = current_reward
            best_iteration_data = env.dirty_data.copy()  # 保存该次迭代的数据
            best_iteration = iteration + 1

        # 更新状态
        state = next_state
    # 输出奖励最高的迭代数据
    print(f"Best Iteration: {best_iteration} with Reward: {best_reward}")
    best_output_filename = f"best_dirty_data_episode_{best_iteration}.csv"
    best_iteration_data.to_csv(best_output_filename, index=False)
    print(f"Best iteration data saved to {best_output_filename}")




if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("RUNNING TIME:"+ str((end - start)/60))
