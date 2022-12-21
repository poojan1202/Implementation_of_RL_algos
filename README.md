# Implementation of RL Algorithms

# Sarsa to Q-Learning

## Monte-Carlo
Implemented Monte-Carlo method on GymMinigrid Envrionment, `MiniGrid-Empty-8x8-v0`.

### Observation Space
- `gen_obs` generates partially observable agent's view (an image)
- For discrete observation, we use `agent_pos`, which returns the grid number at which the agent is present.

### Action Space

| Num | Action |
| ----- | ----- |
| 0  | Turn Left |
| 1  | Turn Right|
| 2  | Move Forward|

### Reward Function
- Reward is 1 when agent reaches goal, else 0

### Hyperparameters
- Gamma
    - 0.9
- Training Episodes
    - 75
- Exploration
    - Epsilon = Epsilon/1.1

### Results
#### Training Reward

<p align="center">
<img src="https://i.imgur.com/ZmLvdXs.png" width="600" height="400" align="Center"> 
</p>

#### Simulation

<p align="center">
<img src="https://i.imgur.com/1SNW6ro.gif" width="500" height="400" align="Center">
</p>

## SARSA-λ and SARSA backward

Implemented SARSA-λ and Backward SARSA method on GymMinigrid Envrionment, `MiniGrid-Empty-8x8-v0` and `MiniGrid-FourRooms-v0`.

### Observation Space
- `gen_obs` generates partially observable agent's view (an image)
- For discrete observation, we use `agent_pos`, which returns the grid number at which the agent is present.

### Action Space

| Num | Action |
| ----- | ----- |
| 0  | Turn Left |
| 1  | Turn Right|
| 2  | Move Forward|

### Reward Function
- Reward is 1 when agent reaches goal, else 0

### Hyperparameters (SARSA-λ)
- Gamma
    - 0.9
- Sarsa Lambda
    - 0.99
- Training Episodes
    - 50
- Exploration
    - Epsilon = Epsilon/1.05

### Results
#### Training Reward

<p align="center">
<img src="https://i.imgur.com/A2CMoZF.png" width="600" height="400" align="Center">
</p>
#### Simulation

<p align="center">
<img src="https://i.imgur.com/Xb5P59S.gif" width="450" height="400" align="Center">
</p>
### Hyperparameters (SARSA Backward)
- Gamma
    - 0.9
- Sarsa Lambda
    - 0.9
- Training Episodes
    - 25
- Exploration
    - Epsilon = Epsilon/1.2

### Results
#### Training Reward

<p align="center">
<img src="https://i.imgur.com/xoLWTBO.png" width="600" height="400" align="Center">
</p>
#### Simulation

<p align="center">
<img src="https://i.imgur.com/rbKsQOx.gif" width="400" height="400" align="Center">
</p>


## Q-Learning

Implemented SARSA-λ and Backward SARSA method on GymMinigrid Envrionment, `MiniGrid-Empty-8x8-v0`.

### Observation Space
- `gen_obs` generates partially observable agent's view (an image)
- For discrete observation, we use `agent_pos`, which returns the grid number at which the agent is present.

### Action Space

| Num | Action |
| ----- | ----- |
| 0  | Turn Left |
| 1  | Turn Right|
| 2  | Move Forward|

### Reward Function
- Reward is 1 when agent reaches goal, else 0

### Hyperparameters
- Gamma
    - Trained agents with 5 different values of gamma
        - 0.9, 0.7, 0.5, 0.3, 0.1
- Training Episodes
    - 150
- Exploration
    - Epsilon = Epsilon/1.1

### Results
#### Training Reward

<p align="center">
<img src="https://i.imgur.com/yKNVz7r.png" width="600" height="400" align="Center">
</p>

#### Steps vs Episodes

<p align="center">
<img src="https://i.imgur.com/Ui8U81l.png" width="600" height="400" align="Center">
</p>

#### Simulation

<p align="center">
<img src="https://i.imgur.com/1SNW6ro.gif" width="500" height="400" align="Center">
</p>

## Deep Q-Learning (DQN)
Implemented DQN on Gym Envrionment, `Gym-CartPole-v0`.
### Observation Space

| Num | Observation | Min | Max |
| ----- | ----- | ----- | -------|
| 0  | Cart Position  | -4.8 | 4.8|
| 1  | Cart Velocity  | -Inf | Inf|
| 2  | Pole Angle  | -0.418 rad(-24 deg) | 0.418 rad(-24 deg)|
| 3  | Pole Angular Velocity  | -Inf | Inf|

### Action Space

| Num | Action |
| ----- | ----- |
| 0  | Push Cart to Left  |
| 1  | Push Cart to Right|

### Reward Function
- Reward is 1 for every step taken, including the termination step

### Hyperparameters
- Network Architecture
    - 4 Linear Layers of dim = [16, 32, 16, 2]
- Optimizer
    - Adam Optimizer
- Learning Rate
    - 0.0001
- Batch Size
    - 128
- Training Episodes
    - 700

### Results

#### Training Reward

<p align="center">
<img src="https://i.imgur.com/EFmIOjm.png" width="600" height="400" align="Center"> 
</p>

#### Simulation

<p align="center">
<img src="https://i.imgur.com/u6c28Ka.gif" width="600" height="400" align="Center">
</p>

## Policy Gradient
Implemented Policy Gradient Method (Actor-Critic) on Gym Envrionment, `Gym-CartPole-v0`.
### Observation Space

The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free end and its angular velocity.
| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(angle)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |


### Action Space

 The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

### Reward Function
- The reward function is a function of theta, angle made by the pendulum.

### Hyperparameters
- Network Architecture
    - Actor
        - 4 Linear Layers of dim = [31,128,32,2]
    - Critic
        - 4 Linear Layers of dim = [31,128,32,1]

- Optimizer
    - Adam Optimizer
- Learning Rate
    - 0.0005
- Batch Size
    - 64
- Training Episodes
    - 1200

### Results
#### Training Reward

<p align="center">
<img src="https://i.imgur.com/9SZnAN6.png" width="600" height="400" align="Center"> 
</p>

#### Simulation

<p align="center">
<img src="https://i.imgur.com/HUcZWfa.gif" width="500" height="400" align="Center">
</p>



