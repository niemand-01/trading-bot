import random

from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


@tf.keras.utils.register_keras_serializable(package="custom")
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (
        tf.abs(error) - clip_delta
    )
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """Stock Trading Bot"""

    def __init__(
        self,
        state_size,
        strategy="t-dqn",
        reset_every=1000,
        pretrained=False,
        model_name=None,
    ):
        self.strategy = strategy

        # agent config
        self.state_size = state_size + 1  # normalized previous days + position state
        self.action_size = 3  # [hold, buy, sell] - position-aware actions
        self.model_name = model_name
        self.long_inventory = []  # Track long positions (FIFO)
        self.short_inventory = []  # Track short positions (FIFO)
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {
            "huber_loss": huber_loss
        }  # important for loading the model from memory
        self.optimizer = Adam(learning_rate=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model"""
        model = Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=128, activation="relu"),
                Dense(units=256, activation="relu"),
                Dense(units=256, activation="relu"),
                Dense(units=128, activation="relu"),
                Dense(units=self.action_size),
            ]
        )

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions"""
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter (open long position)

        action_probs = self.model.predict(state, verbose=0)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory"""
        mini_batch = random.sample(self.memory, batch_size)

        # Batch all states and next_states for efficient GPU processing
        states = np.array([state[0] for state, _, _, _, _ in mini_batch])
        next_states = np.array([next_state[0] for _, _, _, next_state, _ in mini_batch])
        actions = np.array([action for _, action, _, _, _ in mini_batch])
        rewards = np.array([reward for _, _, reward, _, _ in mini_batch])
        dones = np.array([done for _, _, _, _, done in mini_batch])

        # DQN
        if self.strategy == "dqn":
            # Batch predict all next states at once
            next_q_values = self.model.predict(
                next_states, verbose=0, batch_size=batch_size
            )
            # Batch predict all current states at once
            current_q_values = self.model.predict(
                states, verbose=0, batch_size=batch_size
            )

            # Calculate targets in batch
            targets = rewards + (1 - dones) * self.gamma * np.amax(
                next_q_values, axis=1
            )

            # Update q-values for the actions taken
            y_train = current_q_values.copy()
            y_train[np.arange(batch_size), actions] = targets

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            # Batch predict all next states with target model
            next_q_values = self.target_model.predict(
                next_states, verbose=0, batch_size=batch_size
            )
            # Batch predict all current states
            current_q_values = self.model.predict(
                states, verbose=0, batch_size=batch_size
            )

            # Calculate targets in batch
            targets = rewards + (1 - dones) * self.gamma * np.amax(
                next_q_values, axis=1
            )

            # Update q-values for the actions taken
            y_train = current_q_values.copy()
            y_train[np.arange(batch_size), actions] = targets

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            # Batch predict all next states with both models
            next_q_values_main = self.model.predict(
                next_states, verbose=0, batch_size=batch_size
            )
            next_q_values_target = self.target_model.predict(
                next_states, verbose=0, batch_size=batch_size
            )
            # Batch predict all current states
            current_q_values = self.model.predict(
                states, verbose=0, batch_size=batch_size
            )

            # Calculate targets using double DQN (use main network to select, target network to evaluate)
            best_actions = np.argmax(next_q_values_main, axis=1)
            targets = (
                rewards
                + (1 - dones)
                * self.gamma
                * next_q_values_target[np.arange(batch_size), best_actions]
            )

            # Update q-values for the actions taken
            y_train = current_q_values.copy()
            y_train[np.arange(batch_size), actions] = targets

        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        history = self.model.fit(
            states, y_train, epochs=1, verbose=0, batch_size=batch_size
        )
        loss = history.history["loss"][0]

        # increment iteration counter for target network updates
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter += 1

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        """Save model with episode number (for checkpoints)"""
        self.model.save("models/{}_{}".format(self.model_name, episode))
    
    def save_best(self):
        """Save best model to model folder (overwrites previous best model)
        This ensures the model folder always contains the best model from training"""
        self.model.save("models/{}".format(self.model_name))

    def load(self):
        return load_model(
            "models/" + self.model_name, custom_objects=self.custom_objects
        )
