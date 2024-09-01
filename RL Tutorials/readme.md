# Reinforcement Learning Algorithms Exhaustive Lists

### **1. Model-Free Algorithms**

These algorithms do not attempt to model the environment; instead, they learn a policy directly.

* **Value-Based Methods:**
  * **Q-Learning**
    * Deep Q-Learning (DQN)
    * Double DQN
    * Dueling DQN
    * Prioritized Experience Replay DQN
  * **SARSA (State-Action-Reward-State-Action)**
    * Expected SARSA
    * True Online SARSA(λ)
  * **Monte Carlo Methods**
  * **Temporal-Difference (TD) Learning**
    * TD(λ)
    * n-step TD
* **Policy-Based Methods:**
  * **REINFORCE**
  * **Actor-Critic Methods**
    * A2C (Advantage Actor-Critic)
    * A3C (Asynchronous Advantage Actor-Critic)
    * ACER (Actor-Critic with Experience Replay)
    * PPO (Proximal Policy Optimization)
    * DDPG (Deep Deterministic Policy Gradient)
    * TD3 (Twin Delayed DDPG)
    * SAC (Soft Actor-Critic)
* **Policy Gradient Methods:**
  * Vanilla Policy Gradient
  * TRPO (Trust Region Policy Optimization)
  * GAE (Generalized Advantage Estimation)

### **2. Model-Based Algorithms**

These algorithms learn a model of the environment and use it to plan or simulate future actions.

* **Dyna-Q**
* **MPC (Model Predictive Control)**
* **MBPO (Model-Based Policy Optimization)**
* **PETS (Probabilistic Ensembles with Trajectory Sampling)**
* **PlaNet (Planning Networks)**
* **Dreamer**

### **3. Hybrid Algorithms**

These algorithms combine aspects of both model-free and model-based approaches.

* **AlphaGo**
* **AlphaZero**
* **MuZero**

### **4. Distributional Reinforcement Learning**

These methods model the distribution of possible rewards, not just the expected value.

* **C51**
* **Quantile Regression DQN (QR-DQN)**
* **Implicit Quantile Networks (IQN)**
* **FQF (Fully Parameterized Quantile Function)**

### **5. Hierarchical Reinforcement Learning**

These algorithms break down tasks into subtasks.

* **Options Framework**
* **Feudal Reinforcement Learning**
* **HAMs (Hierarchical Abstract Machines)**
* **HIRO (Hierarchical Reinforcement Learning with Off-policy Correction)**

### **6. Multi-Agent Reinforcement Learning (MARL)**

Algorithms designed to handle environments with multiple interacting agents.

* **Independent Q-Learning**
* **MADDPG (Multi-Agent DDPG)**
* **COMA (Counterfactual Multi-Agent Policy Gradients)**
* **QMIX**
* **VDN (Value Decomposition Networks)**

### **7. Exploration-Exploitation Strategies**

These methods focus on balancing the trade-off between exploring new actions and exploiting known rewards.

* **Epsilon-Greedy**
* **UCB (Upper Confidence Bound)**
* **Thompson Sampling**
* **Intrinsic Motivation Approaches**
  * RND (Random Network Distillation)
  * Curiosity-driven Exploration
  * ICM (Intrinsic Curiosity Module)

### **8. Evolutionary Methods**

These algorithms use concepts from evolutionary biology to optimize policies.

* **NEAT (NeuroEvolution of Augmenting Topologies)**
* **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
* **Genetic Algorithms**
* **Evolution Strategies**

### **9. Imitation Learning**

Learning from expert demonstrations rather than trial and error.

* **Behavioral Cloning**
* **GAIL (Generative Adversarial Imitation Learning)**
* **DAgger (Dataset Aggregation)**

### **10. Offline (Batch) Reinforcement Learning**

Algorithms that learn policies from a fixed dataset of interactions.

* **BCQ (Batch-Constrained Q-Learning)**
* **CQL (Conservative Q-Learning)**
* **BEAR (Bootstrapping Error Accumulation Reduction)**
* **BRAC (Batch Reinforcement Learning with Advantage-weighted Conservative Q-Learning)**

### **11. Meta-Reinforcement Learning**

Algorithms that learn to adapt to new tasks quickly.

* **MAML (Model-Agnostic Meta-Learning)**
* **PEARL (Probabilistic Embeddings for Actor-Critic RL)**
* **RL^2 (Reinforcement Learning Squared)**

### **12. Inverse Reinforcement Learning (IRL)**

Learning a reward function based on observed behavior.

* **MaxEnt IRL (Maximum Entropy IRL)**
* **AIRL (Adversarial IRL)**
* **GCL (Guided Cost Learning)**

### **13. Safe Reinforcement Learning**

Algorithms designed to ensure that policies adhere to safety constraints.

* **Constrained Policy Optimization (CPO)**
* **Lagrangian Methods**
* **Shielding**

### **14. Lifelong Reinforcement Learning**

Algorithms that focus on continuous learning over time, adapting to new tasks without forgetting previous ones.

* **EWC (Elastic Weight Consolidation)**
* **Progress & Compress**
* **AGEM (Averaged Gradient Episodic Memory)**

### **15. Neuro-Symbolic Reinforcement Learning**

Combines neural networks with symbolic reasoning for tasks requiring both.

* **Differentiable Inductive Logic Programming**
* **Neuro-Symbolic Concept Learner**

### **16. Miscellaneous & Emerging Methods**

Some other niche or cutting-edge methods.

* **AlphaStar**
* **R2D2 (Recurrent Experience Replay in Distributed Reinforcement Learning)**
* **MuJoCo for continuous control tasks**
