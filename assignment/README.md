# CMPE-258: Advanced Customizations in Deep Learning and Neural Networks

---

## Part 1: Regularization & Data Augmentation

| # | Notebook | Key Concepts | Video |
|---|----------|-------------|-------|
| a | 1a_l1_l2_regularization.ipynb | L1 sparsity, L2 weight decay, Elastic Net, A/B comparison | [▶️ Video](https://drive.google.com/file/d/1u_29XiUVwWL_fDGSrhuGK171f8cugEHq/view?usp=sharing) |
| b | 1b_dropout.ipynb | Standard dropout at 0%, 20%, 50%; overfitting reduction | [▶️ Video](https://drive.google.com/file/d/1MsVYZZOV3PMK8IlFP731cBKxf8ef_5Fw/view?usp=sharing) |
| c | 1c_early_stopping.ipynb | Keras callback, patience, restore_best_weights, manual PyTorch impl | [▶️ Video](https://drive.google.com/file/d/17DSRCJEAhF0NvRECJE50XXNaxaSnZ2mK/view?usp=sharing) |
| d | 1d_monte_carlo_dropout.ipynb | Uncertainty estimation, multiple forward passes, confidence intervals | [▶️ Video](YOUR_VIDEO_LINK_1d) |
| e | 1e_initializations.ipynb | Glorot, He, LeCun, Orthogonal, Zeros; activation distribution analysis | [▶️ Video](YOUR_VIDEO_LINK_1e) |
| f | 1f_batch_normalization.ipynb | BN before activation, convergence speed, mild regularization | [▶️ Video](YOUR_VIDEO_LINK_1f) |
| g | 1g_custom_dropout_regularization.ipynb | AlphaDropout, ConcreteDropout (learned rate), GaussianDropout, custom L1 | [▶️ Video](YOUR_VIDEO_LINK_1g) |
| h | 1h_callbacks_tensorboard.ipynb | ModelCheckpoint, ReduceLR, CSVLogger, custom callback, TB visualization | [▶️ Video](YOUR_VIDEO_LINK_1h) |
| i | 1i_keras_tuner.ipynb | RandomSearch, BayesianOptimization, search space definition | [▶️ Video](YOUR_VIDEO_LINK_1i) |
| j | 1j_kerascv_augmentation.ipynb | RandAugment, RandomFlip, RandomRotation, tf.data pipeline | [▶️ Video](YOUR_VIDEO_LINK_1j) |
| k | 1k_multimodal_augmentation.ipynb | Image (Albumentations), Text (nlpaug), Tabular (noise), TimeSeries | [▶️ Video](YOUR_VIDEO_LINK_1k) |

## Part 2: Advanced Custom Constructs

| # | Notebook | Key Concepts | Video |
|---|----------|-------------|-------|
| i | 2i_custom_lr_scheduler.ipynb | OneCycleScheduler, ExponentialDecay, cosine annealing | [▶️ Video](YOUR_VIDEO_LINK_2i) |
| ii | 2ii_custom_dropout.ipynb | MCAlphaDropout (always-on for SELU networks) | [▶️ Video](YOUR_VIDEO_LINK_2ii) |
| iii | 2iii_custom_normalization.ipynb | MaxNormDense — weight norm clipping | [▶️ Video](YOUR_VIDEO_LINK_2iii) |
| iv | 2iv_tensorboard.ipynb | Custom scalars, histograms, images, overfit gap tracking | [▶️ Video](YOUR_VIDEO_LINK_2iv) |
| v | 2v_custom_loss.ipynb | HuberLoss (function + class), MSE vs MAE vs Huber comparison | [▶️ Video](YOUR_VIDEO_LINK_2v) |
| vi | 2vi_custom_activation_init_reg_constraint.ipynb | Parametric ReLU, Glorot initializer, L1 regularizer, positive weights constraint | [▶️ Video](YOUR_VIDEO_LINK_2vi) |
| vii | 2vii_custom_metric.ipynb | HuberMetric (streaming), R2Score metric | [▶️ Video](YOUR_VIDEO_LINK_2vii) |
| viii | 2viii_custom_layers.ipynb | MyDense, ExponentialLayer, GaussianNoise, LayerNormalization | [▶️ Video](YOUR_VIDEO_LINK_2viii) |
| ix | 2ix_custom_model.ipynb | ResidualBlock, ResidualClassifier with skip connections | [▶️ Video](YOUR_VIDEO_LINK_2ix) |
| x | 2x_custom_optimizer.ipynb | SGD with Momentum from scratch | [▶️ Video](YOUR_VIDEO_LINK_2x) |
| xi | 2xi_custom_training_loop.ipynb | tf.GradientTape, PyTorch manual loop, Fashion MNIST | [▶️ Video](YOUR_VIDEO_LINK_2xi) |
| xii | 2xii_weights_and_biases.ipynb | W&B logging, sweeps, confusion matrix, gradient tracking | [▶️ Video](YOUR_VIDEO_LINK_2xii) |

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
