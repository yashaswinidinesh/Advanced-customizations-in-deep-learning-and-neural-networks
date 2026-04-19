# CMPE-258: Advanced Customizations in Deep Learning and Neural Networks

---

## Part 1: Regularization & Data Augmentation

| # | Notebook | Key Concepts | Video |
|---|----------|-------------|-------|
| a | 1a_l1_l2_regularization.ipynb | L1 sparsity, L2 weight decay, Elastic Net, A/B comparison | [▶️ Video](https://drive.google.com/file/d/1u_29XiUVwWL_fDGSrhuGK171f8cugEHq/view?usp=sharing) |
| b | 1b_dropout.ipynb | Standard dropout at 0%, 20%, 50%; overfitting reduction | [▶️ Video](https://drive.google.com/file/d/1MsVYZZOV3PMK8IlFP731cBKxf8ef_5Fw/view?usp=sharing) |
| c | 1c_early_stopping.ipynb | Keras callback, patience, restore_best_weights, manual PyTorch impl | [▶️ Video](https://drive.google.com/file/d/17DSRCJEAhF0NvRECJE50XXNaxaSnZ2mK/view?usp=sharing) |
| d | 1d_monte_carlo_dropout.ipynb | Uncertainty estimation, multiple forward passes, confidence intervals | [▶️ Video](https://drive.google.com/file/d/1bV2i5ANNLYDZFKQFKMGIm56JBm-TiTbB/view?usp=sharing) |
| e | 1e_initializations.ipynb | Glorot, He, LeCun, Orthogonal, Zeros; activation distribution analysis | [▶️ Video](https://drive.google.com/file/d/1Z_MIds7svrMj3lVDV7IxgXdLwXdB61Ha/view?usp=sharing) |
| f | 1f_batch_normalization.ipynb | BN before activation, convergence speed, mild regularization | [▶️ Video](https://drive.google.com/file/d/1r4CQzGEh87YqeicGXXYfY50b9g8YF6jF/view?usp=sharing) |
| g | 1g_custom_dropout_regularization.ipynb | AlphaDropout, ConcreteDropout (learned rate), GaussianDropout, custom L1 | [▶️ Video](https://drive.google.com/file/d/1CNSzKpTp-lnZx8OpnmLMp3nNRBwShabq/view?usp=sharing) |
| h | 1h_callbacks_tensorboard.ipynb | ModelCheckpoint, ReduceLR, CSVLogger, custom callback, TB visualization | [▶️ Video](https://drive.google.com/file/d/152v-h5heulCq6FscF-hmyKvRwuHly2WI/view?usp=sharing) |
| i | 1i_keras_tuner.ipynb | RandomSearch, BayesianOptimization, search space definition | [▶️ Video](https://drive.google.com/file/d/1BARaQwSqlrIpAaFidWjJQfovTYbqyIes/view?usp=sharing) |
| j | 1j_kerascv_augmentation.ipynb | RandAugment, RandomFlip, RandomRotation, tf.data pipeline | [▶️ Video](https://drive.google.com/file/d/13jg4Jf9V6NIAsmxFo2NHmCbxzPn8yIgY/view?usp=sharing) |
| k | 1k_multimodal_augmentation.ipynb | Image (Albumentations), Text (nlpaug), Tabular (noise), TimeSeries | [▶️ Video](https://drive.google.com/file/d/1ANCQ_gVcmqKZ7_Eu2SuqSkCyVzGWGS4l/view?usp=sharing) |

## Part 2: Advanced Custom Constructs

| # | Notebook | Key Concepts | Video |
|---|----------|-------------|-------|
| i | 2i_custom_lr_scheduler.ipynb | OneCycleScheduler, ExponentialDecay, cosine annealing | [▶️ Video](https://drive.google.com/file/d/1B7-X-j0y-MkPXrtIRTUOnPYk2E3dEEQQ/view?usp=sharing) |
| ii | 2ii_custom_dropout.ipynb | MCAlphaDropout (always-on for SELU networks) | [▶️ Video](https://drive.google.com/file/d/1fD-v1l5XvwDHEtQA69J3xscFcbU4BhGu/view?usp=sharing) |
| iii | 2iii_custom_normalization.ipynb | MaxNormDense — weight norm clipping | [▶️ Video](https://drive.google.com/file/d/1321Pb2xjig7zx0Z1JwSqFTGPhHRVea7p/view?usp=sharing) |
| iv | 2iv_tensorboard.ipynb | Custom scalars, histograms, images, overfit gap tracking | [▶️ Video](https://drive.google.com/file/d/1xUlwm78L7jZ3AWz9UGrSRMzakx86YKEk/view?usp=sharing) |
| v | 2v_custom_loss.ipynb | HuberLoss (function + class), MSE vs MAE vs Huber comparison | [▶️ Video](https://drive.google.com/file/d/1vrwQvE4x26IR9b-EfNhhJFvDN0TIEJqV/view?usp=sharing) |
| vi | 2vi_custom_activation_init_reg_constraint.ipynb | Parametric ReLU, Glorot initializer, L1 regularizer, positive weights constraint | [▶️ Video](https://drive.google.com/file/d/1jwjxchHeysRN7s7EGsM4Cda9c8SnchIe/view?usp=sharing) |
| vii | 2vii_custom_metric.ipynb | HuberMetric (streaming), R2Score metric | [▶️ Video](https://drive.google.com/file/d/1ClzNgbWlIife0AqVsaad88_0N-H21YWx/view?usp=sharing) |
| viii | 2viii_custom_layers.ipynb | MyDense, ExponentialLayer, GaussianNoise, LayerNormalization | [▶️ Video](https://drive.google.com/file/d/1UuInkOMaxY2MHnXgrXJmKbKJ0E2xINnf/view?usp=sharing) |
| ix | 2ix_custom_model.ipynb | ResidualBlock, ResidualClassifier with skip connections | [▶️ Video](https://drive.google.com/file/d/1RLfpza2vLcIJhjmhgvpcDJKcl3r8LX5E/view?usp=sharing) |
| x | 2x_custom_optimizer.ipynb | SGD with Momentum from scratch | [▶️ Video](https://drive.google.com/file/d/1VNnf4KPr_uyPKU--a3DR7q93h29vLUgr/view?usp=sharing) |
| xi | 2xi_custom_training_loop.ipynb | tf.GradientTape, PyTorch manual loop, Fashion MNIST | [▶️ Video](https://drive.google.com/file/d/1RXE616lnarq_xrkwsD2-FCr3k_l9M4Tc/view?usp=sharing) |
| xii | 2xii_weights_and_biases.ipynb | W&B logging, sweeps, confusion matrix, gradient tracking | [▶️ Video](https://drive.google.com/file/d/17rVUK6UmTYXvX270Qx-E-cxrbgfTugsy/view?usp=sharing) |

---

## How to Run
1. Open any `.ipynb` file in Google Colab
2. Select **Runtime → Change runtime type → GPU (T4)**
3. Run all cells sequentially (`Runtime → Run all`)
4. Each notebook is self-contained — all dependencies are installed in the first cell

## References
- Hands-On ML3 (Aurélien Géron)
- Hands-On ML2 (Aurélien Géron)
- KerasCV Documentation
- Awesome Data Augmentation
- AugLy (Facebook Research)
- nlpaug
- Weights & Biases
