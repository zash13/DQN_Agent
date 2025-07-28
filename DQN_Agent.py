from os import stat
from matplotlib.pyplot import cla
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Input
from collections import deque
from enum import Enum


class EpsilonPolicyType(Enum):
    NONE = 0
    DECAY = 1
    SOFTLINEAR = 2


class RewardPolicyType(Enum):
    NONE = 0
    ERM = 1


class UpdateTargetNetworkType(Enum):
    HARD = 0
    SOFT = 1


INFINITY = float("inf")


class ExperienceBuffer:
    def __init__(
        self,
        rewarder: "RewardHelper",
        buffer_size=2000,
        prefer_lower_heuristic=True,
        reward_range=(-10, 20),
    ) -> None:
        self.memory_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.reward_helper = rewarder
        self.prefer_lower_heuristic = prefer_lower_heuristic
        self.reward_range = reward_range

    def store_experience(
        self, current_state, next_state, imm_reward, action, done, heuristic=0
    ):
        imm_reward = self.reward_helper.findReward(
            current_state, imm_reward, heuristic, self.prefer_lower_heuristic
        )
        imm_reward = self.__normalize_reward(self.reward_range, imm_reward)
        self.memory_buffer.append(
            {
                "current_state": current_state,
                "action": action,
                "reward": imm_reward,
                "next_state": next_state,
                "heuristic": heuristic,
                "done": done,
            }
        )

    def sample_batch(self, count):
        if len(self.memory_buffer) < count:
            return None
        batch = random.sample(self.memory_buffer, count)
        states = np.vstack([item["current_state"] for item in batch])
        next_states = np.vstack([item["next_state"] for item in batch])
        rewards = np.array([item["reward"] for item in batch])
        actions = np.array([item["action"] for item in batch])
        dones = np.array([item["done"] for item in batch])
        heuristics = np.array([item["heuristic"] for item in batch])
        return states, next_states, rewards, actions, dones, heuristics

    def __normalize_reward(self, reward_range, reward):
        min_r, max_r = reward_range
        if max_r == min_r:
            return 0.0
        norm = (reward - min_r) / (max_r - min_r)
        return norm * 2 - 1


class QNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.001) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self._model = self._initiate_model()
        self._model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

    def _initiate_model(self):
        return keras.Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=64, activation="relu"),
                Dense(units=32, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def predict(self, states, verbose=0):
        return self._model.predict(states, verbose=verbose)

    def fit(self, states, q_targets, epochs=1, verbose=0):
        history = self._model.fit(states, q_targets, epochs=epochs, verbose=verbose)
        return history.history["loss"][0]

    def set_weights(self, weights):
        self._model.set_weights(weights)
        return True

    def get_weights(self):
        return self._model.get_weights()


# Dueling Q-Network:
# The main difference compared to a standard Q-network is that the network
# splits into two streams near the end: one for the state-value (V) and one for the advantage (A).
# These are then combined using the formula:
# Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
class DuelingQNetwork(QNetwork):
    def __init__(self, state_size, action_size, learning_rate=0.001) -> None:
        super().__init__(state_size, action_size, learning_rate)

    def _initiate_model(self):
        print(self.action_size)
        inputs = keras.Input(shape=(self.state_size,))
        x = keras.layers.Dense(64, activation="relu")(inputs)
        x = keras.layers.Dense(64, activation="relu")(x)
        value = keras.layers.Dense(1, activation=None)(x)
        advantages = keras.layers.Dense(self.action_size, activation=None)(x)
        # combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        mean_advantages = keras.layers.Lambda(
            lambda a: a - keras.backend.mean(a, axis=1, keepdims=True),
            output_shape=(int(self.action_size),),
        )(advantages)
        q_values = keras.layers.Add()([value, mean_advantages])
        return keras.Model(inputs=inputs, outputs=q_values)


# we learnd that agent should balance between exploration and exploitation
# without this section , it seems that agent just try to explor new path
#
class EpsilonPolicy:
    def __init__(
        self,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        policy: EpsilonPolicyType = EpsilonPolicyType.DECAY,
        update_per_episod: bool = True,
    ):
        self.policy = policy
        self.visited_states = {}
        self.epsilon: float = INFINITY
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_per_episod = update_per_episod
        self.previous_episode = 0

    def adjust_epsilon(
        self,
        epsilon,
        episode_count,
        max_episodes=100,
    ):
        self.epsilon = epsilon
        if self.update_per_episod:
            if self.previous_episode == episode_count:
                return self.epsilon
            else:
                self.previous_episode = episode_count
        if self.policy == EpsilonPolicyType.DECAY:
            return self.linear_decay(episode_count, max_episodes)
        elif self.policy == EpsilonPolicyType.SOFTLINEAR:
            return self.soft_linear_decay(episode_count, max_episodes)
        else:
            return self.epsilon

    def linear_decay(self, episode_count, max_episodes):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def soft_linear_decay(self, episode_count, max_episodes=200):
        if self.epsilon > self.epsilon_min:
            target_epsilon = max(
                self.epsilon_min,
                1.0 - (1.0 - self.epsilon_min) * (episode_count / max_episodes),
            )
            self.epsilon = self.epsilon * self.epsilon_decay + target_epsilon * (
                1 - self.epsilon_decay
            )
        return self.epsilon


class RewardHelper:
    def __init__(
        self,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        policy: RewardPolicyType = RewardPolicyType.ERM,
    ):
        self.progress_bonus = progress_bonus
        self.exploration_bonus = exploration_bonus
        self.policy = policy
        self.old_huristic = INFINITY
        self.visited_states = {}

    def findReward(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        if self.policy == RewardPolicyType.ERM:
            return self._ERM(state, reward, heuristic, lowerHuristicBetter)
        elif self.policy == RewardPolicyType.NONE:
            return self._none(reward)
        else:
            return reward

    def _none(self, reward):
        return reward

    def _ERM(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        state_key = tuple(state.flatten()) if state is not None else None
        is_new_state = state_key is not None and state_key not in self.visited_states
        if is_new_state:
            if is_new_state and state_key is not None:
                self.visited_states[state_key] = (
                    heuristic if heuristic is not None else INFINITY
                )

        progress = 0.0
        if heuristic is not None and state_key is not None:
            old_heuristic = self.visited_states[state_key]
            if (
                old_heuristic != INFINITY
                and (heuristic < old_heuristic and lowerHuristicBetter)
                or (heuristic > old_heuristic and not lowerHuristicBetter)
            ):
                progress = self.progress_bonus
                self.visited_states[state_key] = heuristic

        new_reward = reward
        if is_new_state:
            new_reward += self.exploration_bonus
        if progress > 0:
            new_reward += self.progress_bonus
        return new_reward


class DQNAgent:
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        buffer_size=2000,
        batch_size=32,
        gamma=0.99,
        max_episodes=200,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy: EpsilonPolicy = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        prefer_lower_heuristic=True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        reward_range=(0, 0),
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.episode_count = 0
        self.max_episodes = max_episodes

        self.model: "QNetwork" = None
        self.epsilon_policy = epsilon_policy or EpsilonPolicy(
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            policy=EpsilonPolicyType.DECAY,
        )
        self.buffer_helper = ExperienceBuffer(
            RewardHelper(progress_bonus, exploration_bonus, reward_policy),
            buffer_size=buffer_size,
            prefer_lower_heuristic=prefer_lower_heuristic,
            reward_range=reward_range,
        )
        self._define_model(state_size, action_size, learning_rate)

    def _define_model(self, state_size, action_size, learning_rate):
        self.model = QNetwork(state_size, action_size, learning_rate)

    def train(self, episode):
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]
            # q_targets[i, actions[i]] = np.clip(q_targets[i, actions[i]], -10, 10)

        self._update_exploration_rate(episode_count=episode)
        loss = self.model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _update_exploration_rate(self, episode_count):
        self.epsilon = self.epsilon_policy.adjust_epsilon(
            self.epsilon, episode_count=episode_count, max_episodes=self.max_episodes
        )

    def select_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)


class DoubleDQNAgent(DQNAgent):
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        buffer_size=2000,
        batch_size=32,
        gamma=0.99,
        max_episodes=200,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy: EpsilonPolicy = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        prefer_lower_heuristic=True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        update_target_network_method: UpdateTargetNetworkType = UpdateTargetNetworkType.HARD,
        update_factor=0.005,
        target_update_frequency=10,
        reward_range=(0, 0),
    ) -> None:
        super().__init__(
            action_size,
            state_size,
            learning_rate,
            buffer_size,
            batch_size,
            gamma,
            max_episodes,
            epsilon,
            epsilon_min,
            epsilon_decay,
            epsilon_policy,
            reward_policy,
            prefer_lower_heuristic,
            progress_bonus,
            exploration_bonus,
            reward_range,
        )
        self.update_target_network_method = update_target_network_method
        self.online_model = self.model
        self.target_model = None
        self._define_model(state_size, action_size, learning_rate)
        self.previous_episode = 0
        self.update_factor = update_factor
        self.target_model.set_weights(self.online_model.get_weights())
        self.target_update_frequency = target_update_frequency

    def _define_model(self, state_size, action_size, learning_rate):
        self.online_model = QNetwork(state_size, action_size, learning_rate)
        self.target_model = QNetwork(state_size, action_size, learning_rate)

    def train(self, episode):
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data

        # Here's what I understand is happening:
        # We have two models: one acts based on Temporal Difference (TD) learning, and the other (called the target network) behaves more like a Monte Carlo method.
        # Basically, one model is updated frequently (every epoch), while the other is updated less often (at a fixed interval).
        # The frequently updated model is used for action selection, while the target network is used to evaluate Q-values.
        # The TD update equation: Q(s, a) = r + γ * (r + max_a' Q(s', a') - Q(s, a)) becomes:
        # Q_target(s, a) = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        q_current = self.online_model.predict(states, verbose=0)
        q_next_online = self.online_model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                best_action = np.argmax(q_next_online[i])
                q_targets[i, actions[i]] = (
                    rewards[i] + self.gamma * q_next_target[i, best_action]
                )
            else:
                q_targets[i, actions[i]] = rewards[i]
            # q_targets[i, actions[i]] = np.clip(q_targets[i, actions[i]], -10, 10)

        # and here is where target network update base on frequency
        if episode % self.target_update_frequency == 0:
            self._update_target_network()
        self._update_exploration_rate(episode_count=episode)
        loss = self.online_model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _update_target_network(self):
        if self.update_target_network_method == UpdateTargetNetworkType.HARD:
            self.target_model.set_weights(self.online_model.get_weights())
        elif self.update_target_network_method == UpdateTargetNetworkType.SOFT:
            online_weights = self.online_model.get_weights()
            target_weights = self.target_model.get_weights()
            updated_weights = []
            for online_w, target_w in zip(online_weights, target_weights):
                updated_weights.append(
                    self.update_factor * online_w + (1 - self.update_factor) * target_w
                )
            self.target_model.set_weights(updated_weights)

    def select_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.online_model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)


class DuelingDQNAgent(DQNAgent):
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        buffer_size=2000,
        batch_size=32,
        gamma=0.99,
        max_episodes=200,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy: EpsilonPolicy = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        prefer_lower_heuristic=True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        reward_range=(0, 0),
    ) -> None:
        super().__init__(
            action_size,
            state_size,
            learning_rate,
            buffer_size,
            batch_size,
            gamma,
            max_episodes,
            epsilon,
            epsilon_min,
            epsilon_decay,
            epsilon_policy,
            reward_policy,
            prefer_lower_heuristic,
            progress_bonus,
            exploration_bonus,
            reward_range,
        )
        self._define_model(state_size, action_size, learning_rate)

    def _define_model(self, state_size, action_size, learning_rate):
        self.model = DuelingQNetwork(state_size, action_size, learning_rate)

    def train(self, episode):
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]

        self._update_exploration_rate(episode_count=episode)
        loss = self.model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _update_exploration_rate(self, episode_count):
        self.epsilon = self.epsilon_policy.adjust_epsilon(
            self.epsilon, episode_count=episode_count, max_episodes=self.max_episodes
        )

    def select_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)


class DoubleDulelingDQNAgent(DoubleDQNAgent):
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        buffer_size=2000,
        batch_size=32,
        gamma=0.99,
        max_episodes=200,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy: EpsilonPolicy = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        prefer_lower_heuristic=True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        update_target_network_method: UpdateTargetNetworkType = UpdateTargetNetworkType.HARD,
        update_factor=0.005,
        target_update_frequency=10,
        reward_range=(0, 0),
    ) -> None:
        super().__init__(
            action_size,
            state_size,
            learning_rate,
            buffer_size,
            batch_size,
            gamma,
            max_episodes,
            epsilon,
            epsilon_min,
            epsilon_decay,
            epsilon_policy,
            reward_policy,
            prefer_lower_heuristic,
            progress_bonus,
            exploration_bonus,
            update_target_network_method,
            update_factor,
            target_update_frequency,
            reward_range,
        )

    def _define_model(self, state_size, action_size, learning_rate):
        self.online_model = DuelingQNetwork(state_size, action_size, learning_rate)
        self.target_model = DuelingQNetwork(state_size, action_size, learning_rate)

    def train(self, episode):
        self.episode_count += 1
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data
        q_current = self.online_model.predict(states, verbose=0)
        q_next_online = self.online_model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                best_action = np.argmax(q_next_online[i])
                q_targets[i, actions[i]] = (
                    rewards[i] + self.gamma * q_next_target[i, best_action]
                )
            else:
                q_targets[i, actions[i]] = rewards[i]

        self._update_exploration_rate(episode_count=episode)
        if self.episode_count % self.target_update_frequency == 0:
            self._update_target_network()
        loss = self.online_model.fit(states, q_targets, epochs=1, verbose=0)
        return loss
