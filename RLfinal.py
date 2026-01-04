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
        self.original_dirty_data = dirty_data.copy()
        self.dirty_data = dirty_data.copy()
        self.clean_data = clean_data.copy()
        self.n_features = dirty_data.shape[1]
        self.current_step = 0
        self.max_steps = 1
        self.done = False
        self.repair_methods = repair_methods
        self.applied_methods_history = []
        self.mse_history = []
        self.applied_methods_count = np.zeros(len(repair_methods), dtype=np.float32)
        self.iteration_count = 0
        self.original_data = original_data.copy()
        self.quality_history=[]
        self.quality_evaluator=DataQualityEvaluator(self.dirty_data)

    def reset(self):
        self.dirty_data = self.original_dirty_data.copy()
        initial_quality = self.quality_evaluator.evaluate(self.dirty_data)
        print(f"Initial data quality: {initial_quality}")
        self.quality_history = []
        self.quality_history.append(initial_quality)
        self.dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)
        missing_count = self.dirty_data.isnull().sum().sum()
        if missing_count == 0:
            print("警告：数据中没有缺失值。为了训练，将随机引入一些缺失值。")
            mask = np.random.rand(*self.dirty_data.shape) < 0.03
            self.dirty_data = self.dirty_data.mask(mask)
            print(f"人为引入了 {mask.sum()} 个缺失值用于训练")
        self.current_step = 0
        self.done = False
        self.applied_methods_count = np.zeros(len(self.repair_methods), dtype=np.float32)
        state = self.get_state()
        return state

    def reset_dirty_data(self):
        self.dirty_data = self.original_dirty_data.copy()
        initial_quality = self.quality_evaluator.evaluate(self.dirty_data)
        print(f"Initial data quality: {initial_quality}")
        self.quality_history.append(initial_quality)
        self.dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)
        self.current_step = 0
        self.applied_methods_count = np.zeros(len(self.repair_methods), dtype=np.float32)
        self.iteration_count += 1

    def get_state(self):
        missing_ratio = self.dirty_data.isnull().mean().values
        quality = self.quality_history[-1] if self.quality_history else 0
        iteration_normalized = np.array([self.current_step / self.max_steps], dtype=np.float32)
        state = np.concatenate([missing_ratio.astype(np.float32), [quality], iteration_normalized])
        return state

    def step(self, actions):
        self.current_step += 1
        rewards = []
        applied_methods = []
        original_data = self.original_data
        selected_methods = [agent_id for agent_id, action in enumerate(actions) if action > 0.5]
        if not selected_methods:
            next_state = self.get_state()
            reward = self.calculate_reward()
            rewards = [reward for _ in actions]
            return next_state, rewards, self.done, applied_methods
        last_method = selected_methods[-1]
        for agent_id in selected_methods:
            is_last_method = (agent_id == last_method)
            self.apply_fix(agent_id, is_last_method)
            applied_methods.append(agent_id)
        self.applied_methods_history.append(applied_methods.copy())
        next_state = self.get_state()
        reward = self.calculate_reward()
        rewards = [reward for _ in actions]
        self.done = False
        return next_state, rewards, self.done, applied_methods

    def apply_fix(self, agent_id, is_last_method=False):
        data_features = self.dirty_data.copy()
        data_before = data_features.copy()
        numeric_columns = []
        for col in data_features.columns:
            try:
                pd.to_numeric(data_features[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                pass
        if numeric_columns:
            data_features[numeric_columns] = data_features[numeric_columns].replace([np.inf, -np.inf], np.nan)
            for col in numeric_columns:
                data_features[col] = pd.to_numeric(data_features[col], errors='coerce').clip(-1e10, 1e10)
        missing_before = data_features.isnull().sum().sum()
        print(f"Missing values before imputation: {missing_before}")
        missing_mask = data_features.isnull()
        if not missing_mask.any().any():
            print(f"Agent {agent_id}: 没有可修复的缺失值。")
            return
        categorical_columns = [col for col in data_features.columns if col not in numeric_columns]
        if agent_id == 0:
            imputer = IterativeImputer(max_iter=10, random_state=0)
            numeric_only = True
        elif agent_id in [1, 2, 3, 4]:
            n_neighbors = [5, 10, 20, 50][agent_id - 1]
            imputer = KNNImputer(n_neighbors=n_neighbors)
            numeric_only = False
        elif agent_id in [5, 6, 7]:
            n_estimators = [50, 100, 200][agent_id - 5]
            imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=n_estimators), max_iter=20,
                                       random_state=0)
            numeric_only = True
        elif agent_id == 8:
            imputer = SimpleImputer(strategy='mean')
            numeric_only = True
        elif agent_id == 9:
            imputer = SimpleImputer(strategy='most_frequent')
            numeric_only = False
        else:
            return
        if is_last_method:
            print(f"Agent {agent_id}: 正在填充所有剩余的缺失值。")
            if numeric_columns:
                numeric_data = data_features[numeric_columns].copy()
                numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
                if numeric_only:
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                    data_features[numeric_columns] = imputed_numeric
                else:
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(numeric_data),
                        columns=numeric_data.columns,
                        index=numeric_data.index
                    )
                    data_features[numeric_columns] = imputed_numeric
            if categorical_columns:
                if not numeric_only and agent_id in [1, 2, 3, 4]:
                    for col in categorical_columns:
                        if data_features[col].isnull().any():
                            non_null_data = data_features[col].dropna()
                            if len(non_null_data) > 0:
                                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                                encoder.fit(non_null_data.values.reshape(-1, 1))
                                all_data = data_features[col].values.reshape(-1, 1)
                                encoded_data = np.zeros((len(all_data), len(encoder.categories_[0])))
                                non_null_indices = ~data_features[col].isnull()
                                if non_null_indices.any():
                                    encoded_non_null = encoder.transform(all_data[non_null_indices])
                                    encoded_data[non_null_indices] = encoded_non_null
                                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                                X_train = np.arange(len(all_data)).reshape(-1, 1)[non_null_indices]
                                y_train = encoded_data[non_null_indices]
                                X_pred = np.arange(len(all_data)).reshape(-1, 1)[~non_null_indices]
                                if len(X_train) > 0 and len(X_pred) > 0 and len(np.unique(y_train, axis=0)) >= min(
                                        n_neighbors, len(y_train)):
                                    knn.fit(X_train, y_train)
                                    y_pred = knn.predict(X_pred)
                                    encoded_data[~non_null_indices] = y_pred
                                    predicted_indices = np.argmax(encoded_data[~non_null_indices], axis=1)
                                    predicted_categories = [encoder.categories_[0][i] for i in predicted_indices]
                                    data_features.loc[~non_null_indices, col] = predicted_categories
                                else:
                                    mode_value = data_features[col].mode().iloc[0] if not data_features[
                                        col].mode().empty else "unknown"
                                    data_features.loc[data_features[col].isnull(), col] = mode_value
                elif agent_id == 9 or numeric_only:
                    for col in categorical_columns:
                        if data_features[col].isnull().any():
                            mode_value = data_features[col].mode().iloc[0] if not data_features[
                                col].mode().empty else "unknown"
                            data_features.loc[data_features[col].isnull(), col] = mode_value
            self.dirty_data = data_features
        else:
            random_mask = data_features.isnull().map(lambda x: np.random.rand() < 0.3 if x else False)
            if numeric_columns:
                numeric_data = data_features[numeric_columns].copy()
                numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
                numeric_random_mask = random_mask[numeric_columns]
                if numeric_only:
                    temp_numeric = numeric_data.copy()
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(temp_numeric),
                        columns=temp_numeric.columns,
                        index=temp_numeric.index
                    )
                    for col in numeric_columns:
                        if col in numeric_random_mask.columns:
                            mask = numeric_random_mask[col]
                            if mask.any():
                                data_features.loc[mask, col] = imputed_numeric.loc[mask, col]
                else:
                    temp_numeric = numeric_data.copy()
                    imputed_numeric = pd.DataFrame(
                        imputer.fit_transform(temp_numeric),
                        columns=temp_numeric.columns,
                        index=temp_numeric.index
                    )
                    for col in numeric_columns:
                        if col in numeric_random_mask.columns:
                            mask = numeric_random_mask[col]
                            if mask.any():
                                data_features.loc[mask, col] = imputed_numeric.loc[mask, col]
            if categorical_columns:
                categorical_random_mask = random_mask[categorical_columns]
                if not numeric_only and agent_id in [1, 2, 3, 4]:
                    for col in categorical_columns:
                        if col in categorical_random_mask.columns and categorical_random_mask[col].any():
                            non_null_data = data_features[col].dropna()
                            if len(non_null_data) > 0:
                                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                                encoder.fit(non_null_data.values.reshape(-1, 1))
                                all_data = data_features[col].values.reshape(-1, 1)
                                encoded_data = np.zeros((len(all_data), len(encoder.categories_[0])))
                                non_null_indices = ~data_features[col].isnull()
                                if non_null_indices.any():
                                    encoded_non_null = encoder.transform(all_data[non_null_indices])
                                    encoded_data[non_null_indices] = encoded_non_null
                                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                                X_train = np.arange(len(all_data)).reshape(-1, 1)[non_null_indices]
                                y_train = encoded_data[non_null_indices]
                                fill_indices = categorical_random_mask[col].values
                                X_pred = np.arange(len(all_data)).reshape(-1, 1)[fill_indices]
                                if len(X_train) > 0 and len(X_pred) > 0 and len(np.unique(y_train, axis=0)) >= min(
                                        n_neighbors, len(y_train)):
                                    knn.fit(X_train, y_train)
                                    y_pred = knn.predict(X_pred)
                                    encoded_data[fill_indices] = y_pred
                                    predicted_indices = np.argmax(encoded_data[fill_indices], axis=1)
                                    predicted_categories = [encoder.categories_[0][i] for i in predicted_indices]
                                    data_features.loc[fill_indices, col] = predicted_categories
                                else:
                                    mode_value = data_features[col].mode().iloc[0] if not data_features[
                                        col].mode().empty else "unknown"
                                    data_features.loc[fill_indices, col] = mode_value
                elif agent_id == 9 or numeric_only:
                    for col in categorical_columns:
                        if col in categorical_random_mask.columns and categorical_random_mask[col].any():
                            mode_value = data_features[col].mode().iloc[0] if not data_features[
                                col].mode().empty else "unknown"
                            data_features.loc[categorical_random_mask[col], col] = mode_value
            self.dirty_data = data_features
            print(f"Agent {agent_id} applied fix ({self.repair_methods[agent_id]}). Data changed in {random_mask.sum().sum()} cells.")
        missing_after = self.dirty_data.isnull().sum().sum()
        if missing_after == 0:
            print("Missing values after imputation: 0")
        self.applied_methods_count[agent_id] += 1

    def calculate_reward(self):
        previous_quality = self.quality_history[-1] if self.quality_history else None
        current_quality = self.quality_evaluator.evaluate(self.dirty_data)
        self.quality_history.append(current_quality)
        if previous_quality is None:
            return 0
        reward = (current_quality - previous_quality) * 100
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
        samples = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = samples
        states_tensor = torch.FloatTensor(states.reshape(self.batch_size, -1))
        actions_tensor = torch.FloatTensor(actions.reshape(self.batch_size, -1))
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states.reshape(self.batch_size, -1))
        for agent in self.agents:
            next_actions = [self.agents[i].target_act(next_states[:, i, :]) for i in range(self.num_agents)]
            next_actions_tensor = torch.FloatTensor(np.concatenate(next_actions, axis=1))
            target_inputs = torch.cat([next_states_tensor, next_actions_tensor], dim=1)
            with torch.no_grad():
                target_Q = rewards_tensor[:, agent.agent_id].unsqueeze(1) + self.gamma * agent.target_critic(target_inputs)
            current_inputs = torch.cat([states_tensor, actions_tensor], dim=1)
            current_Q = agent.critic(current_inputs)
            critic_loss = nn.MSELoss()(current_Q, target_Q)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            agent_state = torch.FloatTensor(states[:, agent.agent_id, :])
            agent_action = agent.actor(agent_state)
            other_actions = []
            for i in range(self.num_agents):
                if i == agent.agent_id:
                    other_actions.append(agent_action)
                else:
                    other_actions.append(torch.FloatTensor(actions[:, i, :]))
            all_actions = torch.cat(other_actions, dim=1)
            actor_inputs = torch.cat([states_tensor, all_actions], dim=1)
            actor_loss = -agent.critic(actor_inputs).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            agent.update_target_network(self.tau)

def handle_full_nan_columns(dirty_data, original_data):
    cols_with_all_missing = dirty_data.columns[(dirty_data == -999999.0).all()]
    for col in cols_with_all_missing:
        original_col = pd.to_numeric(original_data[col], errors='coerce')
        dirty_data[col] = original_col.apply(
            lambda x: x if isinstance(x, (int, float)) else np.nan
        )
        dirty_data[col] = dirty_data[col].replace(np.nan,-999999.0)
    return dirty_data

class DataQualityEvaluator:
    def __init__(self, initial_data):
        self.initial_data = initial_data.copy()
        self.initial_data_processed = self.initial_data.replace([-999999.0, "-999999.0"], np.nan)
        self.compute_baseline_statistics()

    def compute_baseline_statistics(self):
        self.numerical_columns = []
        self.categorical_columns = []
        self.numerical_stats = {}
        self.categorical_stats = {}
        for col in self.initial_data_processed.columns:
            numeric_values = pd.to_numeric(self.initial_data_processed[col], errors='coerce')
            non_nan_ratio = numeric_values.notna().mean()
            if non_nan_ratio > 0.3:
                self.numerical_columns.append(col)
                valid_values = numeric_values.dropna()
                if len(valid_values) > 0:
                    median = valid_values.median()
                    q1 = valid_values.quantile(0.25)
                    q3 = valid_values.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    normal_values = valid_values[(valid_values >= lower_bound) & (valid_values <= upper_bound)]
                    self.numerical_stats[col] = {
                        'median': median,
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'mean': normal_values.mean() if len(normal_values) > 0 else median,
                        'std': normal_values.std() if len(normal_values) > 0 else iqr / 1.35,
                        'skewness': self.calculate_skewness(normal_values),
                        'kurtosis': self.calculate_kurtosis(normal_values),
                        'valid_count': len(valid_values),
                        'distribution_samples': normal_values.sample(min(100, len(normal_values))).values if len(normal_values) > 0 else []
                    }
            else:
                self.categorical_columns.append(col)
                valid_values = self.initial_data_processed[col].dropna()
                if len(valid_values) > 0:
                    value_counts = valid_values.value_counts(normalize=True)
                    self.categorical_stats[col] = {
                        'value_counts': value_counts,
                        'entropy': self.calculate_entropy(value_counts),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                        'unique_ratio': len(value_counts) / len(valid_values),
                        'valid_count': len(valid_values)
                    }
        if len(self.numerical_columns) > 1:
            numeric_data = self.initial_data_processed[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
            self.correlation_matrix = numeric_data.corr(method='spearman').fillna(0)
        else:
            self.correlation_matrix = pd.DataFrame()

    def calculate_entropy(self, probabilities):
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def calculate_skewness(self, values):
        if len(values) < 3:
            return 0
        return values.skew()

    def calculate_kurtosis(self, values):
        if len(values) < 4:
            return 0
        return values.kurtosis()

    def evaluate(self, data):
        data_copy = data.copy()
        data_copy = data_copy.replace([-999999.0, "-999999.0"], np.nan)
        distribution_score = self.evaluate_distribution(data_copy)
        relationship_score =  self.evaluate_relationships(data_copy)
        anomaly_score = self.evaluate_anomalies(data_copy)
        print(f"Distribution similarity: {distribution_score:.4f}")
        print(f"Relationship preservation: {relationship_score:.4f}")
        print(f"Anomaly score: {anomaly_score:.4f}")
        weights = [0.4, 0.3, 0.3]
        scores = [distribution_score, relationship_score, anomaly_score]
        valid_scores = []
        valid_weights = []
        for i, score in enumerate(scores):
            if not np.isnan(score):
                valid_scores.append(score)
                valid_weights.append(weights[i])
        if not valid_scores:
            print("Warning: No valid quality scores calculated, using default value 0.5")
            return 0.5
        valid_weights = [w / sum(valid_weights) for w in valid_weights]
        overall_score = sum(s * w for s, w in zip(valid_scores, valid_weights))
        print(f"Overall quality score: {overall_score:.4f}")
        return overall_score

    def evaluate_distribution(self, data):
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 1.0
            means = numeric_data.mean(skipna=True)
            stds = numeric_data.std(skipna=True)
            if means.isna().any() or stds.isna().any():
                valid_cols = ~(means.isna() | stds.isna())
                if not valid_cols.any():
                    return 1.0
                means = means[valid_cols]
                stds = stds[valid_cols]
                numeric_data = numeric_data[valid_cols.index[valid_cols]]
            z_scores = np.abs((numeric_data - means) / stds.replace(0, 1))
            outlier_threshold = 3.0
            outlier_ratio = (z_scores > outlier_threshold).mean().mean()
            distribution_score = 1.0 - min(1.0, outlier_ratio * 10)
            return distribution_score
        except Exception as e:
            print(f"Error in distribution evaluation: {e}")
            return 1.0

    def evaluate_relationships(self, data):
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty or numeric_data.shape[1] < 2:
                return 1.0
            numeric_data_clean = numeric_data.dropna()
            if len(numeric_data_clean) < 2:
                return 1.0
            try:
                corr_matrix = numeric_data_clean.corr()
                if corr_matrix.isna().all().all():
                    return 1.0
                upper_tri = np.triu(corr_matrix.values, k=1)
                strong_corr_ratio = np.sum(np.abs(upper_tri) > 0.5) / max(1, (upper_tri.size - np.sum(np.isnan(upper_tri))))
                relationship_score = min(1.0, strong_corr_ratio * 2)
                return relationship_score
            except Exception as e:
                print(f"Error in correlation calculation: {e}")
                return 1.0
        except Exception as e:
            print(f"Error in relationship evaluation: {e}")
            return 1.0

    def evaluate_anomalies(self, data):
        try:
            missing_ratio = data.isna().mean().mean()
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                q1 = numeric_data.quantile(0.25)
                q3 = numeric_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).mean().mean()
                anomaly_ratio = (missing_ratio + outliers) / 2
            else:
                anomaly_ratio = missing_ratio
            anomaly_score = 1.0 - min(1.0, anomaly_ratio * 5)
            return anomaly_score
        except Exception as e:
            print(f"Error in anomaly evaluation: {e}")
            return 1.0

def main():
    dirty_data = pd.read_csv('reconstructed_data.csv')
    clean_data = pd.read_csv('beers_clean.csv')
    original_data = pd.read_csv('beers_clean.csv')
    if 'reconstruction_error' in dirty_data.columns:
        dirty_data_features = dirty_data.drop(columns=['reconstruction_error'])
    else:
        dirty_data_features = dirty_data
    if dirty_data_features.columns.tolist() != clean_data.columns.tolist():
        print("Error: Features in dirty_data and clean_data do not match.")
        print("dirty_data features:", dirty_data_features.columns.tolist())
        print("clean_data features:", clean_data.columns.tolist())
        return
    repair_methods = ['EM', 'KNN5', 'KNN10', 'KNN20', 'KNN50', 'MissForest50', 'MissForest100', 'MissForest200',
                    'Mean Imputation', 'Mode Imputation']
    num_agents = len(repair_methods)
    state_size = dirty_data.shape[1] + 1 + 1
    action_size = 1
    dirty_data = handle_full_nan_columns(dirty_data,original_data)
    env = DataCleaningEnv(dirty_data, clean_data, repair_methods, original_data)
    dirty_data.replace([-999999.0, "-999999.0"], np.nan, inplace=True)
    maddpg = MADDPG(num_agents, state_size, action_size)
    num_episodes = 200
    max_steps = env.max_steps
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}: Starting training...")
        state = env.reset()
        episode_rewards = 0
        for step in range(max_steps):
            actions = []
            noise_scale = max(0.1, 0.99 ** episode)
            for agent in maddpg.agents:
                action = agent.act(state.reshape(1, -1), noise_scale=noise_scale)
                actions.append(action)
            next_state, rewards, done, _ = env.step(actions)
            states = np.array([state for _ in range(num_agents)])
            next_states = np.array([next_state for _ in range(num_agents)])
            actions_array = np.array(actions).reshape(num_agents, -1)
            rewards_array = np.array(rewards)
            maddpg.memory.push(states, actions_array, rewards_array, next_states)
            if len(maddpg.memory) > maddpg.batch_size:
                maddpg.update()
            state = next_state
            episode_rewards += np.mean(rewards_array)
            if done:
                break
        print(f'Episode {episode + 1}/{num_episodes}')
        print(f'Rewards: {episode_rewards}')
    print("\nEvaluating Agents...")
    max_iterations = 10
    env.mse_history = []
    env.applied_methods_history = []
    env.applied_methods_count = np.zeros(len(repair_methods), dtype=np.float32)
    best_reward = float('-inf')
    best_iteration_data = None
    best_iteration =  0
    best_applied_methods = None
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Starting...")
        env.reset_dirty_data()
        state = env.get_state()
        final_actions = []
        for agent in maddpg.agents:
            student_action = agent.act(state.reshape(1, -1), noise_scale=0.5)
            final_actions.append(student_action)
        next_state, rewards, done, applied_methods = env.step(final_actions)
        method_names = [repair_methods[agent_id] for agent_id in applied_methods]
        print(f"Iteration {iteration + 1}:")
        print(f"Selected Repair Methods: {method_names}")
        print(f"Data quality after repair: {env.quality_history[-1]}")
        print(f"Reward: {rewards[0]}")
        current_reward = rewards[0]
        if current_reward > best_reward:
            best_reward = current_reward
            best_iteration_data = env.dirty_data.copy()
            best_iteration = iteration + 1
            best_applied_methods = method_names
    print(f"\nBest Iteration: {best_iteration} with Reward: {best_reward}")
    print(f"Best Methods Applied: {best_applied_methods}")
    best_output_filename = f"best_dirty_data_iteration_{best_iteration}.csv"
    best_iteration_data.to_csv(best_output_filename, index=False)
    print(f"Best iteration data saved to {best_output_filename}")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("RUNNING TIME:" + str((end - start) / 60))