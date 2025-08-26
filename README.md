# GRPO-style LLM Fine-Tuning with vLLM and DeepSpeed

This repository contains a Python script for fine-tuning a Large Language Model (LLM) using a **GRPO-style (Generalized Reinforcement Policy Optimization) algorithm**. The implementation leverages a powerful distributed architecture, combining **vLLM** for high-throughput inference (rollouts) and **DeepSpeed** for efficient, distributed training of the policy model. The entire system is orchestrated using **Ray**.

The primary goal is to enhance the model's ability to solve specific tasks by rewarding correct completions, as demonstrated here with the **GSM8K mathematical reasoning dataset**.

## 🚀 Features

  * **GRPO-like Policy Optimization**: Implements a core proximal policy optimization loop to stably train the LLM from rewards.
  * **High-Throughput Generation**: Uses **vLLM** as a detached, high-performance inference engine to generate experience rollouts quickly.
  * **Efficient Distributed Training**: Employs **DeepSpeed** (ZeRO Stage 1) to manage memory and scale the training process for the policy model.
  * **Asynchronous-like Architecture**: Uses **Ray actors** to separate the concerns of data generation (`VLLMActor`) and model training (`PolicyModelActor`), allowing them to operate concurrently.
  * **Custom Weight Synchronization**: Includes a mechanism to broadcast updated model weights from the training actor back to the vLLM inference engine, keeping the generation policy up-to-date.
  * **Modular and Configurable**: Key parameters like model name, batch sizes, learning rate, and more are centralized for easy experimentation.

-----

## 🏗️ Architecture

The system operates using two main types of Ray actors that communicate with each other. This separation allows for specialized optimizations—vLLM for generation and DeepSpeed for training.

1.  **Data Sampling**: The main loop pulls a batch of prompts from the GSM8K dataset.
2.  **Rollout Generation**: The prompts are sent to the `VLLMActor`. This actor, powered by vLLM, generates multiple (`N_ROLLOUTS`) completions for each prompt.
3.  **Reward Calculation**: The main script evaluates the generated completions. For GSM8K, a simple reward function is used: a reward of **1.0** is given if the extracted numerical answer is correct, and **0.0** otherwise. Advantages are calculated by normalizing rewards (subtracting the mean).
4.  **Policy Update**: The experience (prompts, completions, old log-probabilities, and advantages) is passed to the `PolicyModelActor`. This actor uses DeepSpeed to perform several epochs of GRPO updates on the experience batch.
5.  **Weight Synchronization**: Periodically, the updated weights from the `PolicyModelActor` are broadcast back to the `VLLMActor` to ensure the inference engine is using the latest policy.
