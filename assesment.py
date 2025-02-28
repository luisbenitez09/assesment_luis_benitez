"""
Aproach: Q-Learning algorithm to solve the assesment problem
"""

import random
import numpy as np

# Defining the bidimensional space
rewards = np.array([
    [0, 0, 0, 0,],
    [0, 0, 0, 0,],
    [0, 0, -0.04, 0,],
    [0, 0, 1, -0.1,],
])

# Defining the possible actions

actions = ['up', 'down', 'right', 'left', 'nop']
actions_indices = {action: i for i, action in enumerate(actions)}

# Defining the initial parameters

alpha = 0.1 # leargnig rate
gamma = 0.9 # discount factor
epsilon = 0.1 # exploration factor
n_episodes = 800

# We define the Q-values matrix

Q = np.zeros((4, 4, len(actions)))

# We define the function to get the next state with the current state and the action taken
def get_next_state(future_state, action_taken):
    x, y = future_state
    if action_taken == 'up' and x > 0:
        return (x - 1, y)
    elif action_taken == 'down' and x < 3:
        return (x + 1, y)
    elif action_taken == 'left' and y > 0:
        return (x, y - 1)
    elif action_taken == 'right' and y < 3:
        return (x, y + 1)
    else:
        return future_state  # nop movement


# Choose the action based on the epsilon-greedy policy
def choose_action(actual_state):
    if random.uniform(0, 1) < epsilon:  # Explore
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[actual_state[0], actual_state[1]])] # Exploit

# Q-Learning algorithm
for episode in range(n_episodes):
    # We set a random initial state within the grid
    state = (random.randint(0, 3), random.randint(0, 3))
    while True:
        action = choose_action(state) # Choose an action using epsilon-greedy
        next_state = get_next_state(state, action) # Get the next state
        reward = rewards[next_state[0], next_state[1]] # Get the possible new reward
        best_next_action = np.argmax(Q[next_state[0], next_state[1]]) # Get the best action for the next state
        # Update the Q-value for the state-action pair
        Q[state[0], state[1], actions_indices[action]] += alpha * (reward + gamma * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], actions_indices[action]])
        state = next_state # Update the state
        if state == (3, 2):  # Goal state
            break

# Print the learned Q-values
print("Learned Q-values:")
print(Q)

# Print the learned policy
print("Learned policy:")
policy = np.argmax(Q, axis=2)
for i in range(4):
    print(policy[i])
