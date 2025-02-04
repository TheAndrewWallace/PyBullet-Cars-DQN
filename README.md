# PyBullet Cars Reinforcement Learning DQN
This project includes two husky robot cars in PyBullet controlled by a single DQN neural network, and aims to swap places with the other car. It is a simple implementation for experimenting with reinforcement learning.

Some interesting analysis of the results is that the final model did not require any punishment for collisions, likely because this implicitly prevented them achieving the reward anyway. The model was further improved by removing the additional reward given for reducing distance to the target following the cars learning how to get there.

See below for a video demonstrating the simulation!

https://www.youtube.com/watch?v=FVBYpEpgDEU
