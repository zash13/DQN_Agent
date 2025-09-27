## Project Source

### Articles about Q-Learning

- [Deep Q-Networks: Theory and Implementation](https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67)
- [A Guide to Deep Q-Networks (DQNs)](https://medium.com/@jamesnorthfield2001/a-guide-to-deep-q-networks-dqns-806f6f4805f4)
- [Reinforcement Learning with TensorFlow Agents (Official Tutorial)](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Epsilon Gready (Balancing exploration and exploitation in episodic reinforcement learning)](https://www.sciencedirect.com/science/article/abs/pii/S0957417423013039)
- [Reinforcement Learning, Optimal Policy Search with MDP (Optimal Policy)](https://www.geeksforgeeks.org/deep-learning/temporal-difference-td-learning/)
- Double Deep Q-Networks:
- [Temporal Difference (TD) learning \_ Why 𝛾maxQ(s',a')-Q(s,a)
  (TD)](https://www.geeksforgeeks.org/deep-learning/temporal-difference-td-learning/)

### Research Papers

- [A Theoretical Analysis of Deep Q-Learning (arXiv:1901.00137)](https://arxiv.org/abs/1901.00137)
- [Playing Atari with Deep Reinforcement Learning (arXiv:1312.5602)](https://arxiv.org/abs/1312.5602)
- [Deep Reinforcement Learning with Double Q-learning (arXiv:1509.06461)](https://arxiv.org/abs/1509.06461)

- [Dueling Network Architectures for Deep Reinforcement Learning (arXiv:1511.06581)](https://arxiv.org/abs/1511.06581)

### Youtube Refferences :

- [ Dueling Deep Q Learning with Tensorflow 2 & Keras | Full Tutorial for Beginners (Dueling QNetwork)](https://www.youtube.com/watch?v=CoePrz751lg&t=1214s)

### Experiments and Development

- `DQN_Agent.py`:

  - [Originally written within this repository. (DQN Maze)](https://github.com/13biti/DQN_Maze)
  - Check commit messages for a timeline of development, experiments, and agent evolution in that repository .

  - Later refactored into a standalone submodule named `DQN_Agent`.

### Call for Collaboration

- i am actively seeking contributions from other programmers to improve the DQN agent . Specific areas where we need help include:

  1. Reward Policies: Designing more effective reward structures to improve the agent's learning efficiency and performance.

  2. Epsilon Decay Strategy: Optimizing the epsilon-greedy exploration strategy to balance exploration and exploitation more effectively.

  3. Chain Inheritance in DQN: Implementing or improving chain inheritance mechanisms to make the DQN architecture more modular and extensible.

New Ideas: We are open to innovative approaches, such as novel network architectures, advanced exploration techniques, or integration with other reinforcement learning methods.

If you're interested in contributing, please feel free to:

- Fork the repository and submit pull requests with your improvements.
- Share your ideas or suggestions via issues or discussions in the DQN Maze repository.
- Collaborate on refactoring, testing, or adding new features to the DQN_Agent module.

Let's work together to build a more powerful and versatile DQN agent! Join us in this exciting journey of reinforcement learning exploration.
