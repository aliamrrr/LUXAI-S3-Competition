# AI Bots for LuxAI Season 3 Competition 🤖⚽

Welcome to the repository for the AI bots developed for the LuxAI Season 3 competition! The goal of this competition is to create and train AI bots to play a novel multi-agent 1v1 game against other agents. This repository includes the various reinforcement learning techniques used to train our models and an evaluation framework for assessing their performance.

## Overview 📝

In this competition, we developed multiple techniques to train our bots, including an adaptive training strategy and pre-trained models. The following techniques are implemented:

- **Adaptative DQN**: A variant of Deep Q-Network (DQN) that trains directly against an opponent, adapting its strategy during training based on the adversary's behavior. 💡
- **Pretrained DQN**: A DQN model that has been pre-trained on initial data to accelerate learning. 🚀
- **Dueling DQN with PRP (Pretrained Reward Prediction)**: A combination of Dueling DQN with Pretrained Reward Prediction, which helps the agent better handle uncertain environments by using a value and advantage function to separate out state-action values. 🏆
- **QMix Learning**: A multi-agent reinforcement learning approach for cooperative environments. 🤝

An evaluation framework (`LUXAI_EVALUATION.ipynb`) is provided to benchmark these models and compare their performance. 📊

## Techniques 💻

### 1. **Adaptative DQN** 🎮
The **Adaptative DQN** algorithm is a variant of DQN that directly trains against an opponent. This method allows the agent to adapt its strategy by interacting with a real adversary during training, providing a more dynamic learning environment and helping the agent generalize better to unseen opponents.

### 2. **Pretrained DQN** 🚀
The **Pretrained DQN** model leverages pre-trained weights to jumpstart the training process.

### 3. **Dueling DQN with PRP (Pretrained Reward Prediction)** 🎯
The **Dueling DQN with PRP** model USES Dueling DQN .

### 4. **QMix Learning** 🧠
The **QMix Learning** algorithm is used in multi-agent environments where agents must collaborate. It is designed for cooperative gameplay, allowing agents to work together while learning independently.

## Evaluation 📈

The evaluation framework is provided in the `LUXAI_EVALUATION.ipynb` notebook. This framework can be used to benchmark the performance of the models by evaluating them against each other in various game scenarios. The notebook provides detailed performance metrics.
