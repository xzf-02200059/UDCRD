import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import copy
from collections import deque
import random
import time

class DataCleaningEnv:
    def __init__(self, dirty_data, clean_data, repair_methods,original_data):
        """
        初始化环境。

        参数：
        - dirty_data：带有错误的数据集（DataFrame）
        - clean_data：对应的干净数据集（DataFrame），用于计算奖励
        - repair_methods：修复方法的名称列表
        """
        self.original_dirty_data = dirty_data.copy()
        self.dirty_data = dirty_data.copy()
        self.clean_data = clean_data.copy()
        self.n_features = dirty_data.shape[1]
        self.current_step = 0
        self.max_steps = 1  # 每个回合只进行一次修复
        self.done = False
        self.repair_methods = repair_methods
        self.applied_methods_history = []  # 用于记录每次迭代中选择的修复方法
        self.mse_history = []  # 用于记录每次迭代的 MSE
        self.applied_methods_count = np.zeros(len(repair_methods), dtype=np.float32)  # 记录修复方法的应用次数
        self.iteration_count = 0
        self.original_data = original_data.copy()
        self.quality_history=[]

        self.quality_evaluator=DataQualityEvaluator(self.dirty_data)

    def reset(self):
        """
        重置环境，开始新的一轮。
        """
        self.dirty_data = self.original_dirty_data.copy()

        # 计算初始数据质量
        initial_quality = self.quality_evaluator.evaluate(self.dirty_data)
        print(f"Initial data quality: {initial_quality}")
        self.quality_history = []
        self.quality_history.append(initial_quality)

        # 计算带有 -999999 的 MSE（mse_before）
        # mse_before = self.calculate_mse(include_invalid_values=True)
        # print(f"Initial MSE (with -999999): {mse_before}")
        # self.mse_history = []
        # self.mse_history.append(mse_before)

        #self.dirty_data.replace(-999999.0, np.nan, inplace=True)
        self.dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)
        # 检查是否有缺失值
        missing_count = self.dirty_data.isnull().sum().sum()
        if missing_count == 0:
            print("警告：数据中没有缺失值。为了训练，将随机引入一些缺失值。")
            # 随机将1-5%的数据设为NaN
            mask = np.random.rand(*self.dirty_data.shape) < 0.03
            self.dirty_data = self.dirty_data.mask(mask)
            print(f"人为引入了 {mask.sum()} 个缺失值用于训练")
        self.current_step = 0
        self.done = False
        # 不重置 applied_methods_history，以便在所有迭代中累积记录
        #self.mse_history = []
        self.applied_methods_count = np.zeros(len(self.repair_methods), dtype=np.float32)
        state = self.get_state()
        # 计算初始的 MSE
        #mse = self.calculate_mse()
        #self.mse_history.append(mse)
        return state

    def reset_dirty_data(self):
        """
        重置脏数据为初始状态，但不重置其他累积状态。
        """
        self.dirty_data = self.original_dirty_data.copy()

        # 计算初始数据质量
        initial_quality = self.quality_evaluator.evaluate(self.dirty_data)
        print(f"Initial data quality: {initial_quality}")
        self.quality_history.append(initial_quality)

        # 计算带有 -999999 的 MSE（mse_before）
        # mse_before = self.calculate_mse(include_invalid_values=True)
        # print(f"Initial MSE (with -999999): {mse_before}")
        # self.mse_history.append(mse_before)

        #self.dirty_data.replace(-999999.0, np.nan, inplace=True)
        self.dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)

        self.current_step = 0
        self.applied_methods_count = np.zeros(len(self.repair_methods), dtype=np.float32)
        # 重新计算初始的 MSE
        #mse = self.calculate_mse()
        #self.mse_history = [mse]
        #self.mse_history.append(mse)
        # 更新迭代计数器
        self.iteration_count += 1

    def get_state(self):
        """
        获取当前状态：返回每个特征的缺失值比例、当前的 MSE 和迭代次数。
        """
        missing_ratio = self.dirty_data.isnull().mean().values
        quality = self.quality_history[-1] if self.quality_history else 0
        #mse = self.calculate_mse()
        iteration_normalized = np.array([self.current_step / self.max_steps], dtype=np.float32)
        #iteration_normalized = np.array([self.iteration_count / 100.0], dtype=np.float32)
        state = np.concatenate([missing_ratio.astype(np.float32), [quality], iteration_normalized])
        return state

    def step(self, actions):
        """
        执行智能体的动作，更新环境状态。

        参数：
        - actions：所有智能体的动作列表。

        返回：
        - next_state：下一状态。
        - rewards：所有智能体的奖励列表。
        - done：环境是否结束。
        """
        self.current_step += 1
        rewards = []
        applied_methods = []
        original_data = self.original_data

        # 生成修复概率分布
        #repair_probabilities = np.random.dirichlet(np.ones(len(actions)))
        # 获取所有被选择的方法
        selected_methods = [agent_id for agent_id, action in enumerate(actions) if action > 0.5]

        # 如果没有被选择的方法，直接返回
        if not selected_methods:
            next_state = self.get_state()
            reward = self.calculate_reward()
            rewards = [reward for _ in actions]
            return next_state, rewards, self.done, applied_methods

        # 标记最后一个方法
        last_method = selected_methods[-1]

        # 应用各智能体的修复方法
        for agent_id in selected_methods:
            is_last_method = (agent_id == last_method)
            self.apply_fix(agent_id, is_last_method)
            applied_methods.append(agent_id)

        # 保存本次迭代的修复方法
        self.applied_methods_history.append(applied_methods.copy())

        next_state = self.get_state()
        reward = self.calculate_reward()
        rewards = [reward for _ in actions]

        # 不终止环境，允许多次迭代
        self.done = False

        return next_state, rewards, self.done, applied_methods

    def apply_fix(self, agent_id, is_last_method=False):
        """
        根据智能体ID，应用对应的修复方法，并更新已应用的修复方法计数。
        """
        data_features = self.dirty_data.copy()
        data_before = data_features.copy()
        original_data = self.original_data

        # 替换 inf 和过大值，但只对数值型列进行操作
        # 首先识别数值型列
        numeric_columns = []
        for col in data_features.columns:
            # 尝试将列转换为数值型
            try:
                pd.to_numeric(data_features[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                # 如果转换失败，说明列包含非数值数据
                pass

        # 只对数值型列进行替换和裁剪操作
        if numeric_columns:
            data_features[numeric_columns] = data_features[numeric_columns].replace([np.inf, -np.inf], np.nan)
            for col in numeric_columns:
                # 单独对每列进行裁剪，避免混合类型问题
                data_features[col] = pd.to_numeric(data_features[col], errors='coerce').clip(-1e10, 1e10)

        # 检查缺失值数量
        missing_before = data_features.isnull().sum().sum()
        print(f"Missing values before imputation: {missing_before}")

        # 获取缺失值位置
        missing_mask = data_features.isnull()

        # 如果没有缺失值，输出信息并返回
        if not missing_mask.any().any():
            print(f"Agent {agent_id}: 没有可修复的缺失值。")
            return

        # 准备用于填充的数据
        # 对于分类型数据，我们需要特殊处理
        categorical_columns = [col for col in data_features.columns if col not in numeric_columns]

        # 创建一个可以处理混合数据类型的填充器
        if agent_id == 0:
            # EM 修复（使用 IterativeImputer + 高斯过程）- 只用于数值型数据
            imputer = IterativeImputer(max_iter=10, random_state=0)
            # 这个方法只适用于数值型数据
            numeric_only = True
        elif agent_id in [1, 2, 3, 4]:
            # KNN 修复 - 可以处理数值型数据，但需要对分类数据进行特殊处理
            n_neighbors = [5, 10, 20, 50][agent_id - 1]
            imputer = KNNImputer(n_neighbors=n_neighbors)
            # KNN可以处理数值型数据，但需要对分类数据进行编码
            numeric_only = False
        elif agent_id in [5, 6, 7]:
            # MissForest 修复（使用随机森林）- 只用于数值型数据
            n_estimators = [50, 100, 200][agent_id - 5]
            imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=n_estimators), max_iter=20,
                                       random_state=0)
            # 这个方法只适用于数值型数据
            numeric_only = True
        elif agent_id == 8:
            # 填充平均值 - 只用于数值型数据
            imputer = SimpleImputer(strategy='mean')
            # 这个方法只适用于数值型数据
            numeric_only = True
        elif agent_id == 9:
            # 填充众数 - 适用于所有数据类型
            imputer = SimpleImputer(strategy='most_frequent')
            # 这个方法可以处理所有数据类型
            numeric_only = False
        else:
            return

        # 最后一个方法：修复所有剩余的缺失值
        if is_last_method:
            print(f"Agent {agent_id}: 正在填充所有剩余的缺失值。")

            # 处理数值型数据
            if numeric_columns:
                numeric_data = data_features[numeric_columns].copy()
                numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

                # 应用填充器
                if numeric_only:
                    # 只对数值型数据应用此填充器
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                    data_features[numeric_columns] = imputed_numeric
                else:
                    # 对于KNN和众数填充，可以处理数值型数据
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                    data_features[numeric_columns] = imputed_numeric

            # 处理分类型数据
            if categorical_columns:
                if not numeric_only and agent_id in [1, 2, 3, 4]:
                    # 对于KNN方法，需要对分类数据进行编码处理
                    for col in categorical_columns:
                        if data_features[col].isnull().any():
                            # 对分类数据进行独热编码
                            non_null_data = data_features[col].dropna()
                            if len(non_null_data) > 0:
                                # 创建编码器
                                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                                # 拟合非空数据
                                encoder.fit(non_null_data.values.reshape(-1, 1))

                                # 对所有数据进行编码
                                all_data = data_features[col].values.reshape(-1, 1)
                                encoded_data = np.zeros((len(all_data), len(encoder.categories_[0])))

                                # 只编码非空值
                                non_null_indices = ~data_features[col].isnull()
                                if non_null_indices.any():
                                    encoded_non_null = encoder.transform(all_data[non_null_indices])
                                    encoded_data[non_null_indices] = encoded_non_null

                                # 使用KNN填充编码后的数据
                                knn = KNeighborsClassifier(n_neighbors=n_neighbors)

                                # 找出非空的索引作为训练数据
                                X_train = np.arange(len(all_data)).reshape(-1, 1)[non_null_indices]
                                y_train = encoded_data[non_null_indices]

                                # 找出空值的索引作为预测数据
                                X_pred = np.arange(len(all_data)).reshape(-1, 1)[~non_null_indices]

                                if len(X_train) > 0 and len(X_pred) > 0 and len(np.unique(y_train, axis=0)) >= min(
                                        n_neighbors, len(y_train)):
                                    # 训练KNN模型
                                    knn.fit(X_train, y_train)
                                    # 预测空值
                                    y_pred = knn.predict(X_pred)
                                    # 将预测结果填回编码数据
                                    encoded_data[~non_null_indices] = y_pred

                                    # 将编码数据转换回分类
                                    # 对每行找出最大值的索引
                                    predicted_indices = np.argmax(encoded_data[~non_null_indices], axis=1)
                                    # 使用索引获取对应的分类值
                                    predicted_categories = [encoder.categories_[0][i] for i in predicted_indices]
                                    # 填充原始数据
                                    data_features.loc[~non_null_indices, col] = predicted_categories
                                else:
                                    # 如果无法使用KNN，则使用众数填充
                                    mode_value = data_features[col].mode().iloc[0] if not data_features[
                                        col].mode().empty else "unknown"
                                    data_features.loc[data_features[col].isnull(), col] = mode_value
                elif agent_id == 9 or numeric_only:
                    # 对于众数填充或其他方法，直接使用众数
                    for col in categorical_columns:
                        if data_features[col].isnull().any():
                            mode_value = data_features[col].mode().iloc[0] if not data_features[
                                col].mode().empty else "unknown"
                            data_features.loc[data_features[col].isnull(), col] = mode_value

            self.dirty_data = data_features
        else:
            # 随机修复部分缺失值
            random_mask = data_features.isnull().map(lambda x: np.random.rand() < 0.3 if x else False)

            # 处理数值型数据
            if numeric_columns:
                numeric_data = data_features[numeric_columns].copy()
                numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

                # 创建随机掩码的子集，只针对数值型列
                numeric_random_mask = random_mask[numeric_columns]

                # 应用填充器
                if numeric_only:
                    # 只对数值型数据应用此填充器
                    temp_numeric = numeric_data.copy()
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(temp_numeric),
                        columns=temp_numeric.columns,
                        index=temp_numeric.index
                    )

                    # 只更新随机选择的缺失值
                    for col in numeric_columns:
                        if col in numeric_random_mask.columns:
                            mask = numeric_random_mask[col]
                            if mask.any():
                                data_features.loc[mask, col] = imputed_numeric.loc[mask, col]
                else:
                    # 对于KNN和众数填充，可以处理数值型数据
                    temp_numeric = numeric_data.copy()
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(temp_numeric),
                        columns=temp_numeric.columns,
                        index=temp_numeric.index
                    )

                    # 只更新随机选择的缺失值
                    for col in numeric_columns:
                        if col in numeric_random_mask.columns:
                            mask = numeric_random_mask[col]
                            if mask.any():
                                data_features.loc[mask, col] = imputed_numeric.loc[mask, col]

            # 处理分类型数据
            if categorical_columns:
                categorical_random_mask = random_mask[categorical_columns]

                if not numeric_only and agent_id in [1, 2, 3, 4]:
                    # 对于KNN方法，需要对分类数据进行编码处理
                    for col in categorical_columns:
                        if col in categorical_random_mask.columns and categorical_random_mask[col].any():
                            # 对分类数据进行独热编码
                            non_null_data = data_features[col].dropna()
                            if len(non_null_data) > 0:
                                # 创建编码器
                                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                                # 拟合非空数据
                                encoder.fit(non_null_data.values.reshape(-1, 1))

                                # 对所有数据进行编码
                                all_data = data_features[col].values.reshape(-1, 1)
                                encoded_data = np.zeros((len(all_data), len(encoder.categories_[0])))

                                # 只编码非空值
                                non_null_indices = ~data_features[col].isnull()
                                if non_null_indices.any():
                                    encoded_non_null = encoder.transform(all_data[non_null_indices])
                                    encoded_data[non_null_indices] = encoded_non_null

                                # 使用KNN填充编码后的数据
                                knn = KNeighborsClassifier(n_neighbors=n_neighbors)

                                # 找出非空的索引作为训练数据
                                X_train = np.arange(len(all_data)).reshape(-1, 1)[non_null_indices]
                                y_train = encoded_data[non_null_indices]

                                # 找出要填充的索引（随机选择的缺失值）
                                fill_indices = categorical_random_mask[col].values
                                X_pred = np.arange(len(all_data)).reshape(-1, 1)[fill_indices]

                                if len(X_train) > 0 and len(X_pred) > 0 and len(np.unique(y_train, axis=0)) >= min(
                                        n_neighbors, len(y_train)):
                                    # 训练KNN模型
                                    knn.fit(X_train, y_train)
                                    # 预测空值
                                    y_pred = knn.predict(X_pred)
                                    # 将预测结果填回编码数据
                                    encoded_data[fill_indices] = y_pred

                                    # 将编码数据转换回分类
                                    # 对每行找出最大值的索引
                                    predicted_indices = np.argmax(encoded_data[fill_indices], axis=1)
                                    # 使用索引获取对应的分类值
                                    predicted_categories = [encoder.categories_[0][i] for i in predicted_indices]
                                    # 填充原始数据
                                    data_features.loc[fill_indices, col] = predicted_categories
                                else:
                                    # 如果无法使用KNN，则使用众数填充
                                    mode_value = data_features[col].mode().iloc[0] if not data_features[
                                        col].mode().empty else "unknown"
                                    data_features.loc[fill_indices, col] = mode_value
                elif agent_id == 9 or numeric_only:
                    # 对于众数填充或其他方法，直接使用众数
                    for col in categorical_columns:
                        if col in categorical_random_mask.columns and categorical_random_mask[col].any():
                            mode_value = data_features[col].mode().iloc[0] if not data_features[
                                col].mode().empty else "unknown"
                            data_features.loc[categorical_random_mask[col], col] = mode_value

            self.dirty_data = data_features
            # Print result
            print(
                f"Agent {agent_id} applied fix ({self.repair_methods[agent_id]}). Data changed in {random_mask.sum().sum()} cells.")

        # 在所有修复完成后检查缺失值是否全部修复
        missing_after = self.dirty_data.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after imputation: 0")

        # 更新已应用的修复方法计数
        self.applied_methods_count[agent_id] += 1


    def calculate_reward(self):
        """
        计算奖励：使用每次迭代中 MSE 的减少量。
        """
        #clean_data_features = self.clean_data
        # mse_before 是初始的 MSE
        # mse_before = self.mse_history[-1]
        #
        # # mse_after 是应用修复方法后的 MSE
        # mse_after = self.calculate_mse()
        #
        # reward = (mse_before - mse_after) / mse_before * 100 # 放大奖励值
        # print(f'mse_before: {mse_before}, mse_after: {mse_after}, reward: {reward}')
        #
        # # 更新 mse_history
        # self.mse_history.append(mse_after)

        # 获取上一次的数据质量指标
        previous_quality = self.quality_history[-1] if self.quality_history else None

        # 计算当前的数据质量指标
        current_quality = self.quality_evaluator.evaluate(self.dirty_data)

        # 将当前质量指标添加到历史记录
        self.quality_history.append(current_quality)

        # 如果是第一次计算，返回0作为奖励
        if previous_quality is None:
            return 0

        # 计算奖励：质量指标的改善程度
        reward = (current_quality - previous_quality) * 100  # 放大奖励值

        print(f'Previous quality: {previous_quality}, Current quality: {current_quality}, Reward: {reward}')

        return reward


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

        self.update_target_network(1.0)  # 初始化 target 网络

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
        # 确保 state 是 NumPy 数组，并且形状为 (1, state_size)
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        state = torch.FloatTensor(state)
        action = self.actor(state)
        action = action.detach().numpy()
        noise = noise_scale * np.random.randn(*action.shape)
        action = np.clip(action + noise, 0.0, 1.0)  # 确保动作值在 [0, 1] 之间
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

from collections import deque
import random

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

class MADDPG:
    def __init__(self, num_agents, state_size, action_size):
        self.agents = [Agent(state_size, action_size, agent_id, num_agents) for agent_id in range(num_agents)]
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(1000000)

        self.batch_size = 64
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

def handle_full_nan_columns(dirty_data, original_data):
    """
    处理全为空的列：将这些列的数值型数据替换为原始数据中的相应列，
    但如果原始数据中有乱码（如字符串），则保留为 NaN。
    """
    # 获取全为 -999999 的列（可能会变为 NaN）
    cols_with_all_missing = dirty_data.columns[(dirty_data == -999999.0).all()]

    for col in cols_with_all_missing:
        # 获取原始数据列，并且对数值型数据进行替换
        #original_col = original_data[col]
        #print(cols_with_all_missing)
        # 获取原始数据列，并将其转换为数值型（强制转换）
        original_col = pd.to_numeric(original_data[col], errors='coerce')  # 将字符串转换为 NaN，数值保持
        #print(original_col)

        # 使用 pandas 的 `apply` 方法来确保仅替换数值型数据
        dirty_data[col] = original_col.apply(
            lambda x: x if isinstance(x, (int, float)) else np.nan  # 仅替换数值型数据，其他替换为 NaN
        )
        dirty_data[col] = dirty_data[col].replace(np.nan,-999999.0)
        #print(dirty_data[col])
    #print(dirty_data['chord_length'])

    return dirty_data



class StudentAgent:
    def __init__(self, state_size, action_size, agent_id, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.num_agents = num_agents

        # 使用更简单的网络结构
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def build_actor(self):
        # 更简单的网络结构
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),  # 减少隐藏层大小
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Sigmoid()
        )
        return model

    def build_critic(self):
        # 更简单的网络结构
        model = nn.Sequential(
            nn.Linear(self.state_size * self.num_agents + self.action_size * self.num_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        return model

    def act(self, state, noise_scale=0.0):
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


class DistillationMADDPG:
    def __init__(self, num_agents, state_size, action_size):
        # 教师智能体
        self.teacher_agents = [Agent(state_size, action_size, agent_id, num_agents)
                               for agent_id in range(num_agents)]
        # 学生智能体
        self.student_agents = [StudentAgent(state_size, action_size, agent_id, num_agents)
                               for agent_id in range(num_agents)]

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(1000000)

        self.batch_size = 64
        self.gamma = 0.95
        self.tau = 0.01
        #self.temperature = 2.0  # 蒸馏温度参数
        self.temperature = 1.2  # 修改温度参数
        self.alpha = 0.5  # 添加蒸馏权重参数
        self.distillation_lr = 0.001  # 添加专门的蒸馏学习率
        self.min_prob = 1e-6  # 更新：使用更小的最小概率值
        self.max_prob = 1.0 - self.min_prob  # 更新：相应的最大概率值

    def update(self, states_batch=None, actions_batch=None, rewards_batch=None, next_states_batch=None):
        """更新教师智能体"""
        if len(self.memory) < self.batch_size:
            return

        # 如果没有提供批次数据，从记忆中采样
        if states_batch is None:
            samples = self.memory.sample(self.batch_size)
            states_batch, actions_batch, rewards_batch, next_states_batch = samples

        # 将数据转换为张量
        states_tensor = torch.FloatTensor(states_batch.reshape(self.batch_size, -1))
        actions_tensor = torch.FloatTensor(actions_batch.reshape(self.batch_size, -1))
        rewards_tensor = torch.FloatTensor(rewards_batch)
        next_states_tensor = torch.FloatTensor(next_states_batch.reshape(self.batch_size, -1))

        for agent in self.teacher_agents:
            # 更新 Critic 网络
            next_actions = [self.teacher_agents[i].target_act(next_states_batch[:, i, :])
                            for i in range(self.num_agents)]
            next_actions_tensor = torch.FloatTensor(np.concatenate(next_actions, axis=1))

            target_inputs = torch.cat([next_states_tensor, next_actions_tensor], dim=1)
            with torch.no_grad():
                target_Q = rewards_tensor[:, agent.agent_id].unsqueeze(1) + self.gamma * agent.target_critic(
                    target_inputs)

            current_inputs = torch.cat([states_tensor, actions_tensor], dim=1)
            current_Q = agent.critic(current_inputs)

            critic_loss = nn.MSELoss()(current_Q, target_Q)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # 更新 Actor 网络
            agent_state = torch.FloatTensor(states_batch[:, agent.agent_id, :])
            agent_action = agent.actor(agent_state)

            other_actions = []
            for i in range(self.num_agents):
                if i == agent.agent_id:
                    other_actions.append(agent_action)
                else:
                    other_actions.append(torch.FloatTensor(actions_batch[:, i, :]))

            all_actions = torch.cat(other_actions, dim=1)
            actor_inputs = torch.cat([states_tensor, all_actions], dim=1)

            actor_loss = -agent.critic(actor_inputs).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            agent.update_target_network(self.tau)

    def calculate_kl_divergence(self, teacher_actions, student_actions):
        """计算KL散度"""
        # 确保输入是张量
        if not isinstance(teacher_actions, torch.Tensor):
            teacher_actions = torch.FloatTensor(teacher_actions)
        if not isinstance(student_actions, torch.Tensor):
            student_actions = torch.FloatTensor(student_actions)

        # 添加维度，如果需要
        if len(teacher_actions.shape) == 1:
            teacher_actions = teacher_actions.unsqueeze(0)
        if len(student_actions.shape) == 1:
            student_actions = student_actions.unsqueeze(0)

        # 添加平滑处理避免除零
        epsilon = 1e-8
        teacher_actions = torch.clamp(teacher_actions, epsilon, 1 - epsilon)
        student_actions = torch.clamp(student_actions, epsilon, 1 - epsilon)

        # 使用 KL 散度计算
        # kl_div = F.kl_div(
        #     F.log_softmax(student_actions / self.temperature, dim=1),
        #     F.softmax(teacher_actions / self.temperature, dim=1),
        #     reduction='batchmean'
        # ) * (self.temperature ** 2)

        # 直接计算KL散度: KL(P||Q) = p*log(p/q)
        kl_div = torch.sum(teacher_actions * torch.log(teacher_actions / student_actions) +
                           (1 - teacher_actions) * torch.log((1 - teacher_actions) / (1 - student_actions)))

        # 如果结果是inf或nan，返回一个大数
        if torch.isinf(kl_div) or torch.isnan(kl_div):
            return 100.0
        return kl_div.item()

    def distill_policy(self, states):
        """策略蒸馏过程"""
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)

        # 确保状态有正确的维度
        # if len(states.shape) == 1:
        #     states = states.unsqueeze(0)
        if len(states.shape) == 2:
            states = states.reshape(-1, self.state_size)

        distillation_losses = []

        for student, teacher in zip(self.student_agents, self.teacher_agents):
            # 获取教师动作（无梯度）
            with torch.no_grad():
                teacher_actions = teacher.actor(states)

            # 获取学生动作
            student_actions = student.actor(states)

            # 计算蒸馏损失
            distillation_loss = F.kl_div(
                F.log_softmax(student_actions / self.temperature, dim=1),
                F.softmax(teacher_actions / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)


            # 添加任务损失（可选）
            task_loss = F.mse_loss(student_actions, teacher_actions)
            # 动态调整alpha权重
            current_alpha = self.alpha * (1 - task_loss.item())  # MSE越小，越重视蒸馏损失

            # 总损失
            #total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
            total_loss = current_alpha * distillation_loss + (1 - current_alpha) * task_loss

            # 添加L2正则化
            l2_reg = 0.01 * sum(torch.sum(param ** 2) for param in student.actor.parameters())
            total_loss += l2_reg

            # 更新学生网络
            student.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
            student.actor_optimizer.step()

            distillation_losses.append(total_loss.item())

        return np.mean(distillation_losses)


class DataQualityEvaluator:
    """
    无监督数据质量评估器，用于评估数据修复的质量而不依赖于干净的参考数据。
    """

    def __init__(self, initial_data):
        """
        初始化评估器。

        参数：
        - initial_data：初始数据（带有缺失值的脏数据）
        """
        self.initial_data = initial_data.copy()

        # 将-999999替换为NaN以便计算
        self.initial_data_processed = self.initial_data.replace([-999999.0, "-999999.0"], np.nan)

        # 计算基线统计信息
        self.compute_baseline_statistics()

    def compute_baseline_statistics(self):
        """
        从原始脏数据中提取可用的统计信息作为基线。
        """
        # 识别数值型和分类型列
        self.numerical_columns = []
        self.categorical_columns = []
        self.numerical_stats = {}
        self.categorical_stats = {}

        for col in self.initial_data_processed.columns:
            # 尝试将列转换为数值型
            numeric_values = pd.to_numeric(self.initial_data_processed[col], errors='coerce')
            # 计算非NaN值的比例
            non_nan_ratio = numeric_values.notna().mean()

            # 如果大部分非NaN值可以转换为数值，则认为是数值型列
            if non_nan_ratio > 0.3:  # 只要有30%的值是数值就认为是数值型列
                self.numerical_columns.append(col)

                # 计算数值型列的基线统计（使用非NaN值）
                valid_values = numeric_values.dropna()
                if len(valid_values) > 0:
                    # 使用稳健的统计量作为基线
                    median = valid_values.median()
                    q1 = valid_values.quantile(0.25)
                    q3 = valid_values.quantile(0.75)
                    iqr = q3 - q1

                    # 识别并排除极端异常值
                    lower_bound = q1 - 3 * iqr  # 使用更宽松的界限
                    upper_bound = q3 + 3 * iqr
                    normal_values = valid_values[(valid_values >= lower_bound) & (valid_values <= upper_bound)]

                    # 保存基线统计信息
                    self.numerical_stats[col] = {
                        'median': median,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'mean': normal_values.mean() if len(normal_values) > 0 else median,
                        'std': normal_values.std() if len(normal_values) > 0 else iqr / 1.35,  # 估计标准差
                        'skewness': self.calculate_skewness(normal_values),
                        'kurtosis': self.calculate_kurtosis(normal_values),
                        'valid_count': len(valid_values),
                        'distribution_samples': normal_values.sample(min(100, len(normal_values))).values if len(
                            normal_values) > 0 else []
                    }
            else:
                # 分类型列
                self.categorical_columns.append(col)

                # 计算分类型列的基线统计
                valid_values = self.initial_data_processed[col].dropna()
                if len(valid_values) > 0:
                    value_counts = valid_values.value_counts(normalize=True)

                    # 保存基线统计信息
                    self.categorical_stats[col] = {
                        'value_counts': value_counts,
                        'entropy': self.calculate_entropy(value_counts),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                        'unique_ratio': len(value_counts) / len(valid_values),
                        'valid_count': len(valid_values)
                    }

        # 计算列之间的相关性（仅对数值型列）
        if len(self.numerical_columns) > 1:
            numeric_data = self.initial_data_processed[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
            self.correlation_matrix = numeric_data.corr(method='spearman').fillna(0)  # 使用Spearman相关系数，对缺失值更稳健
        else:
            self.correlation_matrix = pd.DataFrame()

    def calculate_entropy(self, probabilities):
        """计算熵"""
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def calculate_skewness(self, values):
        """计算偏度"""
        if len(values) < 3:
            return 0
        return values.skew()

    def calculate_kurtosis(self, values):
        """计算峰度"""
        if len(values) < 4:
            return 0
        return values.kurtosis()

    def evaluate(self, data):
        """
        评估修复后数据的质量。

        参数：
        - data：修复后的数据（应该没有缺失值）

        返回：
        - quality_score：数据质量得分（0-1之间，越高越好）
        """
        # 首先替换特殊值为NaN，以便正确计算
        data_copy = data.copy()
        data_copy = data_copy.replace([-999999.0, "-999999.0"], np.nan)

        # 初始化各项指标
        # distribution_score = 0  # 分布相似度
        # relationship_score = 0  # 关系保持度
        # anomaly_score = 0  # 异常值比例

        # 计算三个质量维度
        distribution_score = self.evaluate_distribution(data_copy)
        relationship_score =  self.evaluate_relationships(data_copy)
        anomaly_score = self.evaluate_anomalies(data_copy)

        # 打印各个维度的分数，便于调试
        print(f"Distribution similarity: {distribution_score:.4f}")
        print(f"Relationship preservation: {relationship_score:.4f}")
        print(f"Anomaly score: {anomaly_score:.4f}")

        # 计算总体质量分数（加权平均）
        weights = [0.4, 0.3, 0.3]  # 可以调整权重

        # 处理可能的nan值
        scores = [distribution_score, relationship_score, anomaly_score]
        valid_scores = []
        valid_weights = []

        for i, score in enumerate(scores):
            if not np.isnan(score):
                valid_scores.append(score)
                valid_weights.append(weights[i])

        # 如果没有有效分数，返回0.5作为默认值
        if not valid_scores:
            print("Warning: No valid quality scores calculated, using default value 0.5")
            return 0.5

        # 重新归一化权重
        valid_weights = [w / sum(valid_weights) for w in valid_weights]

        # 计算加权平均
        overall_score = sum(s * w for s, w in zip(valid_scores, valid_weights))
        print(f"Overall quality score: {overall_score:.4f}")

        return overall_score

    def evaluate_distribution(self, data):
        """
        评估数据分布的合理性
        """
        try:
            # 只考虑数值型列
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return 1.0  # 如果没有数值型列，返回最高分

            # 计算每列的统计特性
            means = numeric_data.mean(skipna=True)
            stds = numeric_data.std(skipna=True)

            # 检查是否有无效值
            if means.isna().any() or stds.isna().any():
                # 只使用有效的列
                valid_cols = ~(means.isna() | stds.isna())
                if not valid_cols.any():
                    return 1.0  # 如果没有有效列，返回最高分

                means = means[valid_cols]
                stds = stds[valid_cols]
                numeric_data = numeric_data[valid_cols.index[valid_cols]]

            # 计算Z分数来检测异常值
            z_scores = np.abs((numeric_data - means) / stds.replace(0, 1))  # 避免除以零

            # 计算超出正常范围的值的比例
            outlier_threshold = 3.0  # 标准差的倍数
            outlier_ratio = (z_scores > outlier_threshold).mean().mean()

            # 转换为0-1分数，0表示全是异常值，1表示没有异常值
            distribution_score = 1.0 - min(1.0, outlier_ratio * 10)  # 乘以10使得效果更明显

            return distribution_score
        except Exception as e:
            print(f"Error in distribution evaluation: {e}")
            return 1.0  # 出错时返回默认值

    def evaluate_relationships(self, data):
        """
        评估数据中变量之间关系的保持程度
        """
        try:
            # 只考虑数值型列
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty or numeric_data.shape[1] < 2:
                return 1.0  # 如果没有足够的数值型列，返回最高分

            # 删除包含NaN的行，以便计算相关性
            numeric_data_clean = numeric_data.dropna()

            if len(numeric_data_clean) < 2:  # 需要至少两行来计算相关性
                return 1.0

            # 计算相关性矩阵
            try:
                corr_matrix = numeric_data_clean.corr()

                # 检查相关性矩阵是否有效
                if corr_matrix.isna().all().all():
                    return 1.0

                # 提取上三角矩阵的相关系数（不包括对角线）
                upper_tri = np.triu(corr_matrix.values, k=1)

                # 计算强相关的比例（绝对值大于0.5的相关系数）
                strong_corr_ratio = np.sum(np.abs(upper_tri) > 0.5) / max(1, (
                        upper_tri.size - np.sum(np.isnan(upper_tri))))

                # 转换为0-1分数
                relationship_score = min(1.0, strong_corr_ratio * 2)  # 乘以2使得效果更明显

                return relationship_score
            except Exception as e:
                print(f"Error in correlation calculation: {e}")
                return 1.0
        except Exception as e:
            print(f"Error in relationship evaluation: {e}")
            return 1.0  # 出错时返回默认值

    def evaluate_anomalies(self, data):
        """
        评估数据中异常值的存在程度
        """
        try:
            # 计算缺失值比例
            missing_ratio = data.isna().mean().mean()

            # 检查数值型列中的异常值
            numeric_data = data.select_dtypes(include=[np.number])

            if not numeric_data.empty:
                # 计算每列的四分位数范围
                q1 = numeric_data.quantile(0.25)
                q3 = numeric_data.quantile(0.75)
                iqr = q3 - q1

                # 设置异常值的界限
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # 计算异常值的比例
                outliers = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).mean().mean()

                # 综合缺失值和异常值
                anomaly_ratio = (missing_ratio + outliers) / 2
            else:
                anomaly_ratio = missing_ratio

            # 转换为0-1分数，0表示全是异常值，1表示没有异常值
            anomaly_score = 1.0 - min(1.0, anomaly_ratio * 5)  # 乘以5使得效果更明显

            return anomaly_score
        except Exception as e:
            print(f"Error in anomaly evaluation: {e}")
            return 1.0  # 出错时返回默认值

    def calculate_js_divergence(self, p, q):
        """
        计算两个概率分布之间的JS散度。

        参数：
        - p, q: 两个概率分布（pandas.Series）

        返回：
        - js_divergence: JS散度值
        """
        # 确保两个分布有相同的类别
        all_categories = set(p.index) | set(q.index)

        # 初始化KL散度
        kl_p_m = 0
        kl_q_m = 0

        # 创建混合分布 m = (p + q) / 2
        m = {}
        for category in all_categories:
            p_val = p.get(category, 0)
            q_val = q.get(category, 0)
            m[category] = (p_val + q_val) / 2

        # 计算 KL(p||m)
        for category in all_categories:
            p_val = p.get(category, 0)
            m_val = m[category]

            if p_val > 0:  # 避免log(0)
                kl_p_m += p_val * np.log2(p_val / max(m_val, 1e-10))

        # 计算 KL(q||m)
        for category in all_categories:
            q_val = q.get(category, 0)
            m_val = m[category]

            if q_val > 0:  # 避免log(0)
                kl_q_m += q_val * np.log2(q_val / max(m_val, 1e-10))

        # JS散度 = (KL(p||m) + KL(q||m)) / 2
        js_divergence = (kl_p_m + kl_q_m) / 2

        return js_divergence

def main():
    # 加载数据
    dirty_data = pd.read_csv('reconstructed_data.csv')
    clean_data = pd.read_csv('beers_clean.csv')
    original_data = pd.read_csv('beers_clean.csv')

    # 确保 `dirty_data` 和 `clean_data` 的特征列一致
    if 'reconstruction_error' in dirty_data.columns:
        dirty_data_features = dirty_data.drop(columns=['reconstruction_error'])
    else:
        dirty_data_features = dirty_data

    # 检查数据形状
    if dirty_data_features.columns.tolist() != clean_data.columns.tolist():
        print("Error: Features in dirty_data and clean_data do not match.")
        print("dirty_data features:", dirty_data_features.columns.tolist())
        print("clean_data features:", clean_data.columns.tolist())
        # 可以在这里进行特征对齐或其他处理
        return

    # 修复方法名称列表
    #repair_methods = ['EM', 'KNN5', 'MissForest50', 'Mean Imputation', 'Mode Imputation']
    repair_methods = ['EM', 'KNN5', 'KNN10', 'KNN20', 'KNN50', 'MissForest50', 'MissForest100', 'MissForest200',
                    'Mean Imputation', 'Mode Imputation']

    # 初始化环境和智能体
    num_agents = len(repair_methods)
    # 状态维度为特征数量 + 当前 MSE + 迭代次数
    state_size = dirty_data.shape[1] + 1 + 1  # 特征缺失值比例 + 当前 MSE + 迭代次数
    action_size = 1  # 动作为是否应用修复方法

    #env = DataCleaningEnv(dirty_data, clean_data, repair_methods,original_data)

    # 先处理那些全为-999999的列
    dirty_data = handle_full_nan_columns(dirty_data,original_data)
    #print(dirty_data['chord_length'])

    env = DataCleaningEnv(dirty_data, clean_data, repair_methods, original_data)

    #dirty_data.replace(-999999.0, np.nan, inplace=True)
    dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)
    #print(dirty_data)
    #print(dirty_data['chord_length'])

    # 初始化带有蒸馏的MADDPG
    maddpg = DistillationMADDPG(num_agents, state_size, action_size)

    # 训练参数
    num_episodes = 200
    max_steps = env.max_steps
    distillation_interval = 5  # 每5个episode进行一次策略蒸馏

    # 开始训练
    # 训练循环
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}: Starting training...")
        state = env.reset()
        episode_rewards = 0
        distillation_losses = []

        for step in range(max_steps):
            # 教师智能体选择动作
            actions = []
            noise_scale = max(0.1, 0.99 ** episode)
            for agent in maddpg.teacher_agents:
                action = agent.act(state.reshape(1, -1), noise_scale=noise_scale)
                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)

            # 存储经验
            states = np.array([state for _ in range(num_agents)])
            next_states = np.array([next_state for _ in range(num_agents)])
            actions_array = np.array(actions).reshape(num_agents, -1)
            rewards_array = np.array(rewards)

            maddpg.memory.push(states, actions_array, rewards_array, next_states)

            # 更新教师智能体和进行策略蒸馏
            if len(maddpg.memory) > maddpg.batch_size:
                # 更新教师智能体
                maddpg.update()

                # 每隔一定步数进行策略蒸馏
                if step % distillation_interval == 0:
                    # 使用当前状态进行策略蒸馏
                    distill_loss = maddpg.distill_policy(states)
                    distillation_losses.append(distill_loss)
                    # 输出当前蒸馏损失
                    if len(distillation_losses) > 0:
                        print(f'Current Distillation Loss: {distill_loss:.6f}')

            state = next_state
            episode_rewards += np.mean(rewards_array)

            if done:
                break

        # 输出训练信息
        print(f'Episode {episode + 1}/{num_episodes}')
        print(f'Rewards: {episode_rewards}')
        if distillation_losses:
            print(f'Average Distillation Loss: {np.mean(distillation_losses)}')

    # 评估阶段：使用学生智能体
    print("\nEvaluating Student Agents...")
    max_iterations = 10
    env.mse_history = []
    env.applied_methods_history = []
    env.applied_methods_count = np.zeros(len(repair_methods), dtype=np.float32)
    best_reward = float('-inf')
    best_iteration_data = None
    best_iteration = 0
    best_applied_methods = None

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Starting...")
        env.reset_dirty_data()
        state = env.get_state()

        # 获取教师和学生的动作
        teacher_actions = []
        student_actions = []
        final_actions = []  # 实际使用的动作

        for teacher, student in zip(maddpg.teacher_agents, maddpg.student_agents):
            # 获取教师动作
            teacher_action = teacher.act(state.reshape(1, -1), noise_scale=noise_scale)
            teacher_actions.append(teacher_action)

            # 获取学生动作
            student_action = student.act(state.reshape(1, -1), noise_scale=noise_scale)
            student_actions.append(student_action)
            final_actions.append(student_action)

        # 计算并打印每个智能体的KL散度
        for i in range(len(teacher_actions)):
            kl_div = maddpg.calculate_kl_divergence(
                teacher_actions[i],
                student_actions[i]
            )
            print(f"Agent {i} KL Divergence: {kl_div:.6f}")
            print(f"Teacher action: {teacher_actions[i]}")
            print(f"Student action: {student_actions[i]}")

        # 执行动作
        next_state, rewards, done, applied_methods = env.step(final_actions)

        # 计算平均KL散度
        avg_kl_div = np.mean([
            maddpg.calculate_kl_divergence(t, s)
            for t, s in zip(teacher_actions, student_actions)
        ])

        method_names = [repair_methods[agent_id] for agent_id in applied_methods]
        print(f"Iteration {iteration + 1}:")
        print(f"Selected Repair Methods: {method_names}")
        #print(f"MSE after repair: {env.mse_history[-1]}")
        print(f"Data quality after repair: {env.quality_history[-1]}")
        print(f"Reward: {rewards[0]}")
        print(f"KL Divergence: {avg_kl_div:.6f}")

        current_reward = rewards[0]
        if current_reward > best_reward:
            best_reward = current_reward
            best_iteration_data = env.dirty_data.copy()
            best_iteration = iteration + 1
            best_applied_methods = method_names

    # 输出最终结果
    print(f"\nBest Iteration: {best_iteration} with Reward: {best_reward}")
    print(f"Best Methods Applied: {best_applied_methods}")
    best_output_filename = f"student_best_dirty_data_iteration_{best_iteration}.csv"
    best_iteration_data.to_csv(best_output_filename, index=False)
    print(f"Best iteration data saved to {best_output_filename}")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("RUNNING TIME:" + str((end - start) / 60))