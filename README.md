# AI Bots for LuxAI Season 3 Competition 🤖

Welcome to the **LUXAI-S3-Competition** repository! This project is dedicated to developing AI bots for the **LuxAI Season 3** competition, a multi-agent 1v1 game where AI agents compete in a novel environment. This repository contains implementations of various reinforcement learning techniques, evaluation frameworks, and detailed documentation to help you understand and contribute to the project.


## 📝 Overview
The goal of this competition is to create and train AI bots capable of competing against other agents. We have implemented several reinforcement learning techniques to train our models, including:

- **Adaptive DQN**: A variant of Deep Q-Network (DQN) that adapts its strategy during training by directly competing against an opponent.
- **Pretrained DQN**: A DQN model pre-trained on initial data to accelerate learning.
- **Dueling DQN with Prioritized Replay Buffer (PRP)**: Combines Dueling DQN with a prioritized replay buffer to improve training efficiency and performance.
- **QMix Learning**: A multi-agent reinforcement learning approach designed for cooperative environments.

Additionally, we provide an evaluation framework (`LUXAI_EVALUATION.ipynb`) to benchmark and compare the performance of these models.


## 💻 Techniques

### 1. Adaptive DQN 🎮
The **Adaptive DQN** algorithm is designed to train directly against an opponent. This approach allows the agent to dynamically adapt its strategy based on the adversary's behavior, providing a more robust and flexible learning process.

### 2. Pretrained DQN 🚀
The **Pretrained DQN** model leverages pre-trained weights to jumpstart the training process, reducing the time required to achieve competitive performance.

### 3. Dueling DQN with PRP 🎯
The **Dueling DQN with PRP** model enhances the standard Dueling DQN by incorporating a prioritized replay buffer. This technique accelerates training by focusing on important experiences, leading to better-tuned parameters and improved performance.

### 4. QMix Learning 🤝
The **QMix Learning** algorithm is tailored for multi-agent environments where agents must collaborate. It enables agents to learn both independently and cooperatively, making it ideal for scenarios requiring teamwork.


## 📈 Evaluation
The evaluation framework is provided in the `LUXAI_EVALUATION.ipynb` notebook. This framework allows you to:

- Benchmark the performance of different models.
- Compare models against each other in various game scenarios.
- Analyze detailed performance metrics.


## 🛠 Installation
To get started with this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aliamrrr/LUXAI-S3-Competition.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd LUXAI-S3-Competition
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## 📂 Project Structure
```
LUXAI-S3-Competition/
├── data/                  # Sample data files
├── models/                # Pre-trained models
├── notebooks/             # Jupyter notebooks for evaluation and training
│   ├── LUXAI_EVALUATION.ipynb
│   └── ...
├── src/                   # Source code for the AI bots
│   ├── adaptive_dqn/      # Adaptive DQN implementation
│   ├── pretrained_dqn/     # Pretrained DQN implementation
│   ├── dueling_dqn_prp/   # Dueling DQN with PRP implementation
│   ├── qmix/              # QMix Learning implementation
│   └── ...
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```


## 🤝 Contributing
We welcome contributions to this project! If you'd like to contribute, please follow these steps:

1. **Fork the repository** and create a new branch for your feature or bug fix.
2. **Commit your changes** with clear and descriptive messages.
3. **Push your branch** to your fork.
4. **Open a pull request** to the main repository.


## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.


## 📧 Contact
For questions or feedback, please contact [aliamrrr](https://github.com/aliamrrr).