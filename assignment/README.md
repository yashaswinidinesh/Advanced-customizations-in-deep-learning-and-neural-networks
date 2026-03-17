# CMPE-258: Advanced Customizations in Deep Learning and Neural Networks

## Assignment Overview
This repository contains **20 Google Colab notebooks** covering advanced deep learning techniques in both **TensorFlow** and **PyTorch**, organized into two parts.

---

## Repository Structure

```
├── README.md
├── part1/                          # Data Augmentation & Regularization Techniques
│   ├── 1a_l1_l2_regularization.ipynb       # L1, L2, Elastic Net regularization
│   ├── 1b_dropout.ipynb                     # Standard Dropout (0%, 20%, 50%)
│   ├── 1c_early_stopping.ipynb              # Early stopping callback
│   ├── 1d_monte_carlo_dropout.ipynb         # MC Dropout for uncertainty estimation
│   ├── 1e_initializations.ipynb             # Glorot, He, LeCun, Orthogonal, etc.
│   ├── 1f_batch_normalization.ipynb         # Batch Norm with A/B comparison
│   ├── 1g_custom_dropout_regularization.ipynb # Alpha Dropout, Concrete Dropout, custom reg
│   ├── 1h_callbacks_tensorboard.ipynb       # Keras callbacks + TensorBoard
│   ├── 1i_keras_tuner.ipynb                 # Hyperparameter tuning with KerasTuner
│   ├── 1j_kerascv_augmentation.ipynb        # KerasCV data augmentation pipeline
│   └── 1k_multimodal_augmentation.ipynb     # Image, Text, Tabular, TimeSeries augmentation
│
├── part2/                          # Advanced Keras & PyTorch Constructs
│   ├── 2i_custom_lr_scheduler.ipynb         # OneCycle, Exponential Decay schedulers
│   ├── 2ii_custom_dropout.ipynb             # MCAlphaDropout implementation
│   ├── 2iii_custom_normalization.ipynb      # MaxNormDense layer
│   ├── 2iv_tensorboard.ipynb                # TensorBoard: scalars, histograms, images
│   ├── 2v_custom_loss.ipynb                 # Huber Loss (function + class)
│   ├── 2vi_custom_activation_init_reg_constraint.ipynb  # Custom activation/init/reg/constraint
│   ├── 2vii_custom_metric.ipynb             # HuberMetric, R2Score streaming metrics
│   ├── 2viii_custom_layers.ipynb            # ExponentialLayer, MyDense, GaussianNoise, LayerNorm
│   ├── 2ix_custom_model.ipynb               # ResidualBlock & ResidualClassifier
│   ├── 2x_custom_optimizer.ipynb            # Custom SGD with Momentum
│   ├── 2xi_custom_training_loop.ipynb       # Custom training loop (Fashion MNIST)
│   └── 2xii_weights_and_biases.ipynb       # W&B integration + sweep
│
└── scripts/
    └── video_script.md                      # 5-minute video walkthrough script
```

---

## Part 1: Regularization & Data Augmentation

| # | Notebook | Key Concepts | Frameworks |
|---|----------|-------------|------------|
| a | L1/L2 Regularization | L1 sparsity, L2 weight decay, Elastic Net, A/B comparison | TF + PyTorch |
| b | Dropout | Standard dropout at 0%, 20%, 50%; overfitting reduction | TF + PyTorch |
| c | Early Stopping | Keras callback, patience, restore_best_weights, manual PyTorch impl | TF + PyTorch |
| d | Monte Carlo Dropout | Uncertainty estimation, multiple forward passes, confidence intervals | TF + PyTorch |
| e | Initializations | Glorot, He, LeCun, Orthogonal, Zeros; activation distribution analysis | TF + PyTorch |
| f | Batch Normalization | BN before activation, convergence speed, mild regularization | TF + PyTorch |
| g | Custom Dropout & Reg | AlphaDropout, ConcreteDropout (learned rate), GaussianDropout, custom L1 | TF + PyTorch |
| h | Callbacks & TensorBoard | ModelCheckpoint, ReduceLR, CSVLogger, custom callback, TB visualization | TensorFlow |
| i | Keras Tuner | RandomSearch, BayesianOptimization, search space definition | TensorFlow |
| j | KerasCV Augmentation | RandAugment, RandomFlip, RandomRotation, tf.data pipeline | TensorFlow |
| k | Multi-Modal Augmentation | Image (Albumentations), Text (nlpaug), Tabular (noise), TimeSeries | TF + Libraries |

## Part 2: Advanced Custom Constructs

| # | Notebook | Key Concepts | Frameworks |
|---|----------|-------------|------------|
| i | Custom LR Scheduler | OneCycleScheduler, ExponentialDecay, cosine annealing | TF + PyTorch |
| ii | Custom Dropout | MCAlphaDropout (always-on for SELU networks) | TF + PyTorch |
| iii | Custom Normalization | MaxNormDense — weight norm clipping | TF + PyTorch |
| iv | TensorBoard | Custom scalars, histograms, images, overfit gap tracking | TensorFlow |
| v | Custom Loss | HuberLoss (function + class), MSE vs MAE vs Huber comparison | TF + PyTorch |
| vi | Custom Components | Parametric ReLU, Glorot initializer, L1 regularizer, positive weights constraint | TF + PyTorch |
| vii | Custom Metric | HuberMetric (streaming), R2Score metric | TensorFlow |
| viii | Custom Layers | MyDense, ExponentialLayer, GaussianNoise, LayerNormalization | TF + PyTorch |
| ix | Custom Model | ResidualBlock, ResidualClassifier with skip connections | TF + PyTorch |
| x | Custom Optimizer | SGD with Momentum from scratch | TF + PyTorch |
| xi | Custom Training Loop | tf.GradientTape, PyTorch manual loop, Fashion MNIST | TF + PyTorch |
| xii | Weights & Biases | W&B logging, sweeps, confusion matrix, gradient tracking | TF + PyTorch |

---

## How to Run
1. Open any `.ipynb` file in Google Colab
2. Select **Runtime → Change runtime type → GPU (T4)**
3. Run all cells sequentially (`Runtime → Run all`)
4. Each notebook is self-contained — all dependencies are installed in the first cell

## Video Walkthrough
See `scripts/video_script.md` for the 5-minute video script explaining each notebook cell by cell.

## References
- [Hands-On ML3 (Aurélien Géron)](https://github.com/ageron/handson-ml3)
- [Hands-On ML2](https://github.com/ageron/handson-ml2)
- [KerasCV Documentation](https://keras.io/keras_cv/)
- [Awesome Data Augmentation](https://brunokrinski.github.io/awesome-data-augmentation/)
- [AugLy (Facebook Research)](https://github.com/facebookresearch/AugLy)
- [nlpaug](https://github.com/makcedward/nlpaug)
- [Weights & Biases](https://wandb.ai/)
