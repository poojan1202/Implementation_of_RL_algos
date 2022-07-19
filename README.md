# Implementation of RL Algorithms

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
| 0  | Push Cart to Right|

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
![](https://i.imgur.com/4c6BwhD.png)

#### Simulation
![](https://i.imgur.com/uv1frRi.gif)



## Policy Gradient
Implemented Policy Gradient Method (Actor Critic) on Gym Envrionment, `Gym-CartPole-v0`.
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
![](https://i.imgur.com/73AlEcu.png)


#### Simulation
![](https://i.imgur.com/oxIGjke.gif)

