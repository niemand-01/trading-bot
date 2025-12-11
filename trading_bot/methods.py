import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.long_inventory = []
    agent.short_inventory = []
    avg_loss = []

    # Calculate position state: 1.0 for long, -1.0 for short, 0.0 for flat
    def get_position_state():
        if len(agent.long_inventory) > 0:
            return 1.0
        elif len(agent.short_inventory) > 0:
            return -1.0
        else:
            return 0.0

    state = get_state(data, 0, window_size + 1, get_position_state())

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        current_price = data[t]
        
        # select an action
        action = agent.act(state)

        # Position-aware actions:
        # BUY (action == 1): Open long if flat, close short if short
        # SELL (action == 2): Close long if long, open short if flat
        # HOLD (action == 0): Do nothing

        if action == 1:  # BUY
            if len(agent.short_inventory) > 0:
                # Close short position (cover short)
                sold_price = agent.short_inventory.pop(0)
                delta = sold_price - current_price  # Profit when price goes down
                reward = delta
                total_profit += delta
            else:
                # Open long position
                agent.long_inventory.append(current_price)

        elif action == 2:  # SELL
            if len(agent.long_inventory) > 0:
                # Close long position
                bought_price = agent.long_inventory.pop(0)
                delta = current_price - bought_price  # Profit when price goes up
                reward = delta
                total_profit += delta
            else:
                # Open short position
                agent.short_inventory.append(current_price)

        # HOLD (action == 0): Do nothing, reward remains 0

        # Calculate next state with updated position
        next_state = get_state(data, t + 1, window_size + 1, get_position_state())
        
        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # Note: Model saving is now handled in train.py based on best validation performance
    # Removed automatic save every 10 episodes to avoid saving non-optimal models

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))



def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.long_inventory = []
    agent.short_inventory = []
    
    # Calculate position state: 1.0 for long, -1.0 for short, 0.0 for flat
    def get_position_state():
        if len(agent.long_inventory) > 0:
            return 1.0
        elif len(agent.short_inventory) > 0:
            return -1.0
        else:
            return 0.0

    state = get_state(data, 0, window_size + 1, get_position_state())

    for t in range(data_length):        
        reward = 0
        current_price = data[t]
        
        # select an action
        action = agent.act(state, is_eval=True)

        # Position-aware actions (same as training):
        # BUY (action == 1): Open long if flat, close short if short
        # SELL (action == 2): Close long if long, open short if flat
        # HOLD (action == 0): Do nothing

        if action == 1:  # BUY
            if len(agent.short_inventory) > 0:
                # Close short position (cover short)
                sold_price = agent.short_inventory.pop(0)
                delta = sold_price - current_price
                reward = delta
                total_profit += delta

                history.append((current_price, "COVER_SHORT"))
                if debug:
                    logging.debug("Cover short at: {} | Position: {}".format(
                        format_currency(current_price), format_position(delta)))
            else:
                # Open long position
                agent.long_inventory.append(current_price)
                history.append((current_price, "BUY"))
                if debug:
                    logging.debug("Buy at: {}".format(format_currency(current_price)))
        
        elif action == 2:  # SELL
            if len(agent.long_inventory) > 0:
                # Close long position
                bought_price = agent.long_inventory.pop(0)
                delta = current_price - bought_price
                reward = delta
                total_profit += delta

                history.append((current_price, "SELL"))
                if debug:
                    logging.debug("Sell at: {} | Position: {}".format(
                        format_currency(current_price), format_position(delta)))
            else:
                # Open short position
                agent.short_inventory.append(current_price)
                history.append((current_price, "SHORT_SELL"))
                if debug:
                    logging.debug("Short sell at: {}".format(format_currency(current_price)))
        
        else:  # HOLD
            history.append((current_price, "HOLD"))

        # Calculate next state with updated position
        next_state = get_state(data, t + 1, window_size + 1, get_position_state())
        
        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
