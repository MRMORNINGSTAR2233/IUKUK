# 📘 Deep Learning Research Paper Implementation Roadmap  
*A complete curated checklist covering NLP, CV, RL, Diffusion, GANs, LLMs, Core Deep Learning, and Vision–Language Models.*

This README organizes ALL research papers you plan to implement.  
Each category is fully listed below in its own section with checkboxes and links.

---

# 📚 Index

Click a category to jump to its section:

- [🧠 Deep Learning — Core Theory](#-deep-learning--core-theory)
- [🔤 Natural Language Processing (NLP)](#-natural-language-processing-nlp)
- [🧩 Computer Vision (CV)](#-computer-vision-cv)
- [🎮 Reinforcement Learning (RL)](#-reinforcement-learning-rl)
- [🌫 Diffusion Models](#-diffusion-models)
- [🎨 Generative Adversarial Networks (GANs)](#-generative-adversarial-networks-gans)
- [🤖 Large Language Models (LLMs)](#-large-language-models-llms)
- [🔗 Vision–Language Models](#-visionlanguage-models)

---

# 🧠 Deep Learning — Core Theory  
(See full list in this section)

- Efficient BackProp (LeCun, 1998)  
- Dropout (2014)  
- BatchNorm (2015)  
- Adam (2015)  
- Xavier Init (2010)  
- ReLU (2010)  
- ResNet (2015)  
- SE-Net (2017)  
- EfficientNet (2019)  
- SimCLR / BYOL / DINO  
- Generalization studies  
➡️ *Full list with links and checkboxes is below in this README.*

---

# 🔤 Natural Language Processing (NLP)

Covers:
- word2vec  
- Seq2Seq  
- Bahdanau Attention  
- Transformers  
- ELMo, ULMFiT  
- BERT → RoBERTa → ALBERT → XLNet  
- BART, T5  
- DPR, RAG  
- GPT, GPT-2  
- Longformer, DeBERTa, Switch Transformer  

➡️ *Scroll to the NLP section for full checklists and links.*

---

# 🧩 Computer Vision (CV)

Includes:
- LeNet-5  
- AlexNet  
- VGG  
- Inception  
- ResNet  
- DenseNet  
- EfficientNet  
- Faster R-CNN, YOLO, Mask R-CNN  
- DETR, ViT, Swin, DINO, SAM  

➡️ *Scroll to CV section for full paper list.*

---

# 🎮 Reinforcement Learning (RL)

Papers:
- DQN  
- Double DQN  
- Dueling DQN  
- Rainbow  
- TRPO  
- PPO  
- A3C  
- DDPG  
- SAC  
- TD3  
- IMPALA  
- MuZero  
- Decision Transformer  
- CQL  

➡️ *Full RL section below.*

---

# 🌫 Diffusion Models

All diffusion model foundations:
- Sohl-Dickstein (2015)  
- DDPM  
- DDIM  
- Improved DDPM  
- Score-based SDE models  
- GLIDE  
- Imagen  
- LDM (Stable Diffusion)  
- Classifier-Free Guidance  
- Diffusion Models Beat GANs  

➡️ *Full detailed list in the Diffusion section.*

---

# 🎨 Generative Adversarial Networks (GANs)

Includes everything:
- GAN  
- cGAN  
- DCGAN  
- InfoGAN  
- Pix2Pix  
- CycleGAN  
- StarGAN  
- Progressive GAN  
- BigGAN  
- StyleGAN → StyleGAN2 → StyleGAN3  
- Spectral Norm  
- TTUR  
- Projection Discriminator  

➡️ *Full GAN section below.*

---

# 🤖 Large Language Models (LLMs)

All major LLMs you provided:
- GPT-3  
- PaLM  
- Chinchilla  
- LLaMA  
- Jurassic-1  
- GLM-130B  
- MT-NLG  
- FLAN  
- UL2  
- Galactica  
- StableLM  

➡️ *See full LLM section in this file.*

---

# 🔗 Vision–Language Models

List includes:
- CLIP  
- ALIGN  
- ViLBERT  

➡️ *Full section further below in this README.*

---

---

# 📄 Full Sections Start Below  
*(All categories with checkboxes and the full paper lists you provided)*

---

👉 **Paste the sections you already have below this line.**  
Your README will now have:

- Master index (what I provided)  
- All detailed sections (the files I generated earlier)  

This creates a clean, navigable single-file README.









# Deep Learning — Core Theory & Applied Milestones  
(Full Checklist With Links)

A complete list of foundational and advanced deep learning theory papers to implement.

---

## 🔧 Optimization, Regularization & Initialization

- [ ] **Efficient BackProp** — LeCun et al., 1998  
      http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

- [ ] **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** — Srivastava et al., 2014  
      https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

- [ ] **Batch Normalization: Accelerating Deep Network Training** — Ioffe & Szegedy, 2015  
      https://arxiv.org/abs/1502.03167

- [ ] **Adam: A Method for Stochastic Optimization** — Kingma & Ba, 2015  
      https://arxiv.org/abs/1412.6980

- [ ] **Understanding the Difficulty of Training Deep Feedforward Neural Networks (Xavier Init)**  
      Glorot & Bengio, 2010  
      http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

- [ ] **Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU)**  
      Krizhevsky, Sutskever & Hinton, 2010  
      https://papers.nips.cc/paper/2010/file/1fb3ac3a8b0d0c3b2c3c4b4b0f1b8b22-Paper.pdf

- [ ] **Layer Normalization** — Ba et al., 2016  
      https://arxiv.org/abs/1607.06450

- [ ] **Weight Decay, Regularization & Generalization** — Early foundational theory (1992)  
      https://link.springer.com/chapter/10.1007/3-540-55719-9_3

- [ ] **Stochastic Depth / DropPath** — Huang et al., 2016  
      https://arxiv.org/abs/1603.09382

- [ ] **Large-Batch Training of Convolutional Networks (Linear Scaling Rule)** — Goyal et al., 2017  
      https://arxiv.org/abs/1706.02677

---

## 🧱 Neural Network Architectures

- [ ] **Network in Network** — Lin et al., 2013  
      https://arxiv.org/abs/1312.4400

- [ ] **Squeeze-and-Excitation Networks (SE-Net)** — Hu et al., 2017  
      https://arxiv.org/abs/1709.01507

- [ ] **ResNet: Deep Residual Learning for Image Recognition** — He et al., 2015  
      https://arxiv.org/abs/1512.03385

- [ ] **EfficientNet: Rethinking Model Scaling** — Tan & Le, 2019  
      https://arxiv.org/abs/1905.11946

---

## 🧠 Transformers & Beyond

- [ ] **Attention Is All You Need** — Vaswani et al., 2017  
      https://arxiv.org/abs/1706.03762

- [ ] **Vision Transformer (ViT)** — Dosovitskiy et al., 2020  
      https://arxiv.org/abs/2010.11929

---

## 🔍 Self-Supervised Learning Foundations

- [ ] **SimCLR** — Chen et al., 2020  
      https://arxiv.org/abs/2002.05709

- [ ] **BYOL (Bootstrap Your Own Latent)** — Grill et al., 2020  
      https://arxiv.org/abs/2006.07733

- [ ] **DINO: Self-Distillation With No Labels** — Caron et al., 2021  
      https://arxiv.org/abs/2104.14294

- [ ] **Understanding Deep Learning Generalization** — Zhang, Neyshabur, etc. (2016–2019)  
      https://arxiv.org/abs/1611.03530

---

## ⭐ Summary

This file contains **every Deep Learning Core Theory paper** you provided —  
*nothing missing, everything organized, all links included.*

# Natural Language Processing (NLP) — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file contains the complete list of NLP research papers you shared — all included, categorized, and linked.

---

## 🔤 Word Embeddings & Early Neural NLP

- [ ] **word2vec — Efficient Estimation of Word Representations in Vector Space**  
      Mikolov et al., 2013  
      https://arxiv.org/abs/1310.4546

---

## 🔁 Sequence Models, Neural Machine Translation & Attention

- [ ] **Sequence-to-Sequence Learning with Neural Networks**  
      Sutskever et al., 2014  
      https://arxiv.org/abs/1409.3215

- [ ] **Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)**  
      Bahdanau et al., 2015  
      https://arxiv.org/abs/1409.0473

- [ ] **Attention Is All You Need (Transformer)**  
      Vaswani et al., 2017  
      https://arxiv.org/abs/1706.03762

---

## 🧠 Contextual Representations & Transfer Learning

- [ ] **ELMo — Deep Contextualized Word Representations**  
      Peters et al., 2018  
      https://arxiv.org/abs/1802.05365

- [ ] **ULMFiT — Universal Language Model Fine-Tuning for Text Classification**  
      Howard & Ruder, 2018  
      https://arxiv.org/abs/1801.06146

---

## 🧩 Transformer Encoder Architectures

- [ ] **BERT — Pre-training of Deep Bidirectional Transformers for Language Understanding**  
      Devlin et al., 2018  
      https://arxiv.org/abs/1810.04805

- [ ] **RoBERTa — A Robustly Optimized BERT Pretraining Approach**  
      Liu et al., 2019  
      https://arxiv.org/abs/1907.11692

- [ ] **XLNet — Generalized Autoregressive Pretraining**  
      Yang et al., 2019  
      https://arxiv.org/abs/1906.08237

- [ ] **ALBERT — A Lite BERT**  
      Lan et al., 2019  
      https://arxiv.org/abs/1909.11942

- [ ] **SpanBERT — Improving Pre-training by Representing and Predicting Spans**  
      Joshi et al., 2020  
      https://arxiv.org/abs/1907.10529

- [ ] **DeBERTa — Decoding-Enhanced BERT with Disentangled Attention**  
      He et al., 2021  
      https://arxiv.org/abs/2006.03654

- [ ] **Longformer — The Long Document Transformer**  
      Beltagy et al., 2020  
      https://arxiv.org/abs/2004.05150

- [ ] **Switch Transformers — Scaling to Trillion Parameter Models**  
      Fedus et al., 2021  
      https://arxiv.org/abs/2101.03961

---

## 📝 Sequence-to-Sequence, Summarization & Retrieval

- [ ] **BART — Denoising Sequence-to-Sequence Pre-training**  
      Lewis et al., 2019  
      https://arxiv.org/abs/1910.13461

- [ ] **T5 — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**  
      Raffel et al., 2020  
      https://arxiv.org/abs/1910.10683

- [ ] **DPR — Dense Passage Retrieval for Open-Domain QA**  
      Karpukhin et al., 2020  
      https://arxiv.org/abs/2004.04906

- [ ] **RAG — Retrieval-Augmented Generation**  
      Lewis et al., 2020  
      https://arxiv.org/abs/2005.11401

---

## ⚡ Efficient Transformer Training & Alternative Objectives

- [ ] **ELECTRA — Pre-training Encoders as Discriminators Rather Than Generators**  
      Clark et al., 2020  
      https://arxiv.org/abs/2003.10555

---

## 🤖 Generative Transformer Language Models (Pre-GPT-3 Era)

- [ ] **Improving Language Understanding by Generative Pre-Training (GPT)**  
      Radford et al., 2018  
      https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

- [ ] **Language Models Are Unsupervised Multitask Learners (GPT-2)**  
      Radford et al., 2019  
      https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

## ⭐ All NLP Papers Confirmed & Included

This file contains **every NLP paper you listed**, including:

✔ word2vec  
✔ Seq2Seq  
✔ Bahdanau Attention  
✔ Transformer  
✔ ELMo  
✔ ULMFiT  
✔ BERT family  
✔ XLNet  
✔ ALBERT  
✔ BART  
✔ T5  
✔ DPR  
✔ RAG  
✔ SpanBERT  
✔ DeBERTa  
✔ Longformer  
✔ Switch Transformers  
✔ GPT + GPT-2  

No papers are missing.

# Computer Vision (CV) — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file includes **every CV research paper** you listed earlier — CNNs, object detection, transformers, segmentation models, and self-supervised vision.

---

## 🏛 Classical CNN Architectures

- [ ] **LeNet-5 (1998)** — *Gradient-Based Learning Applied to Document Recognition*  
      https://ieeexplore.ieee.org/document/726791  
      PDF mirror: https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

- [ ] **AlexNet (2012)** — *ImageNet Classification with Deep CNNs*  
      https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

- [ ] **VGG16 / VGG19 (2014)** — *Very Deep Convolutional Networks*  
      https://arxiv.org/abs/1409.1556

- [ ] **Inception-v1 / GoogLeNet (2015)** — *Going Deeper with Convolutions*  
      https://arxiv.org/abs/1409.4842

- [ ] **ResNet (2015)** — *Deep Residual Learning for Image Recognition*  
      https://arxiv.org/abs/1512.03385

- [ ] **DenseNet (2017)** — *Densely Connected Convolutional Networks*  
      https://arxiv.org/abs/1608.06993

- [ ] **EfficientNet (2019)** — *Rethinking Model Scaling*  
      https://arxiv.org/abs/1905.11946

---

## 🎯 Object Detection

- [ ] **Faster R-CNN (2015)** — *Towards Real-Time Object Detection with RPN*  
      https://arxiv.org/abs/1506.01497

- [ ] **YOLOv1 (2016)** — *You Only Look Once: Unified, Real-Time Object Detection*  
      https://arxiv.org/abs/1506.02640

- [ ] **FPN (2017)** — *Feature Pyramid Networks*  
      https://arxiv.org/abs/1612.03144

- [ ] **Mask R-CNN (2017)** — *Mask R-CNN*  
      https://arxiv.org/abs/1703.06870

- [ ] **PANet (2018)** — *Path Aggregation Network for Instance Segmentation*  
      https://arxiv.org/abs/1803.01534

- [ ] **YOLOv4 (2020)** — *Optimal Speed & Accuracy of Object Detection*  
      https://arxiv.org/abs/2004.10934

---

## 🧩 Vision Transformers & Beyond

- [ ] **Vision Transformer (ViT) — 2020** — *An Image is Worth 16×16 Words*  
      https://arxiv.org/abs/2010.11929

- [ ] **DETR (2020)** — *End-to-End Object Detection with Transformers*  
      https://arxiv.org/abs/2005.12872

- [ ] **Deformable DETR (2020)** — *Deformable Transformers for End-to-End Object Detection*  
      https://arxiv.org/abs/2010.04159

- [ ] **Swin Transformer (2021)** — *Hierarchical Vision Transformer Using Shifted Windows*  
      https://arxiv.org/abs/2103.14030

- [ ] **DINO (2021)** — *Self-Supervised Learning of ViTs by Self-Distillation*  
      https://arxiv.org/abs/2104.14294

- [ ] **SAM (2023)** — *Segment Anything Model*  
      https://arxiv.org/abs/2304.02643

---

## ⭐ Summary

All CV papers from your list are included:

✔ LeNet-5  
✔ AlexNet  
✔ VGG  
✔ Inception  
✔ ResNet  
✔ DenseNet  
✔ EfficientNet  
✔ Faster R-CNN  
✔ YOLOv1  
✔ FPN  
✔ Mask R-CNN  
✔ PANet  
✔ YOLOv4  
✔ ViT  
✔ DETR  
✔ Deformable DETR  
✔ Swin Transformer  
✔ DINO  
✔ SAM  

No omissions.

# Reinforcement Learning (RL) — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file includes ALL RL papers you listed — from DQN → MuZero → Decision Transformer.

---

## 🎮 Deep Q-Learning & Value-Based Methods

- [ ] **DQN — Playing Atari with Deep RL**  
      Mnih et al., 2015  
      https://arxiv.org/abs/1312.5602 (original NIPS workshop version)

- [ ] **Double DQN (DDQN)** — van Hasselt et al., 2016  
      https://arxiv.org/abs/1509.06461

- [ ] **Dueling DQN** — Wang et al., 2016  
      https://arxiv.org/abs/1511.06581

- [ ] **Rainbow DQN** — Hessel et al., 2018  
      https://arxiv.org/abs/1710.02298

---

## 🤖 Actor–Critic & Continuous Control

- [ ] **DDPG — Deep Deterministic Policy Gradient**  
      Lillicrap et al., 2016  
      https://arxiv.org/abs/1509.02971

- [ ] **A3C / Asynchronous Methods for Deep RL**  
      Mnih et al., 2016  
      https://arxiv.org/abs/1602.01783

- [ ] **TRPO — Trust Region Policy Optimization**  
      Schulman et al., 2015  
      https://arxiv.org/abs/1502.05477

- [ ] **PPO — Proximal Policy Optimization**  
      Schulman et al., 2017  
      https://arxiv.org/abs/1707.06347

- [ ] **SAC — Soft Actor-Critic**  
      Haarnoja et al., 2018  
      https://arxiv.org/abs/1801.01290

- [ ] **TD3 — Twin Delayed DDPG**  
      Fujimoto et al., 2018  
      https://arxiv.org/abs/1802.09477

---

## 🧵 Distributed & Large-Scale RL

- [ ] **IMPALA — Scalable Distributed RL**  
      Espeholt et al., 2018  
      https://arxiv.org/abs/1802.01561

---

## 🧠 Model-Based RL & Planning

- [ ] **MuZero — Mastering Atari, Go, Chess & Shogi With a Learned Model**  
      Schrittwieser et al., 2020  
      https://arxiv.org/abs/1911.08265

---

## 📚 Offline RL

- [ ] **CQL — Conservative Q-Learning**  
      Kumar et al., 2020  
      https://arxiv.org/abs/2006.04779

---

## 🧬 Transformer-Based RL

- [ ] **Decision Transformer — RL via Sequence Modeling**  
      Chen et al., 2021  
      https://arxiv.org/abs/2106.01345

---

## ⭐ Summary

This file includes ALL RL papers you provided:

✔ DQN  
✔ Double DQN  
✔ Dueling DQN  
✔ Rainbow  
✔ TRPO  
✔ DDPG  
✔ A3C  
✔ PPO  
✔ SAC  
✔ TD3  
✔ IMPALA  
✔ MuZero  
✔ Decision Transformer  
✔ CQL  

Nothing missing.

# Diffusion Models — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file contains ALL diffusion model papers you listed — from early Sohl-Dickstein diffusion to Stable Diffusion (LDM), Imagen, GLIDE, DDPM, DDIM, SDEs, classifier-free guidance, and more.

---

## 🧱 Foundations of Diffusion Models

- [ ] **Deep Unsupervised Learning using Nonequilibrium Thermodynamics**  
      Sohl-Dickstein et al., 2015  
      https://arxiv.org/abs/1503.03585

---

## 🌫️ Core Diffusion Model Papers

- [ ] **DDPM — Denoising Diffusion Probabilistic Models**  
      Ho et al., 2020  
      https://arxiv.org/abs/2006.11239

- [ ] **DDIM — Denoising Diffusion Implicit Models**  
      Song et al., 2020  
      https://arxiv.org/abs/2010.02502

- [ ] **Improved DDPM — Learning Variances, Class-Conditional Modeling**  
      Nichol & Dhariwal, 2021  
      https://arxiv.org/abs/2102.09672

- [ ] **Diffusion Models Beat GANs** — (Large-Scale Classifier-Guided Diffusion)  
      Dhariwal & Nichol, 2021  
      https://arxiv.org/abs/2105.05233

---

## 📈 Score-Based Models & SDE Framework

- [ ] **Score-Based Generative Modeling through Stochastic Differential Equations (SDEs)**  
      Song et al., 2021  
      https://arxiv.org/abs/2011.13456

---

## 🧭 Conditioning & Guidance Techniques

- [ ] **Classifier-Free Guidance**  
      Ho & Salimans, 2022  
      https://arxiv.org/abs/2207.12598

---

## 🖼️ Text-to-Image Diffusion Models

- [ ] **GLIDE — Hierarchical Text-Conditional Image Generation**  
      Nichol et al., 2021  
      https://arxiv.org/abs/2112.10741

- [ ] **Imagen — Text-to-Image Diffusion Models**  
      Saharia et al., 2022  
      https://arxiv.org/abs/2205.11487

- [ ] **Latent Diffusion Models (LDM) — Stable Diffusion**  
      Rombach et al., 2022  
      https://arxiv.org/abs/2112.10752

---

## ⭐ Summary

This file includes **all diffusion papers you requested**:

✔ Sohl-Dickstein (2015)  
✔ DDPM  
✔ Improved DDPM  
✔ DDIM  
✔ Score-based SDE models  
✔ GLIDE  
✔ Imagen  
✔ LDM / Stable Diffusion  
✔ Classifier-Free Guidance  
✔ Diffusion Models Beat GANs  

Nothing missing.

# Generative Adversarial Networks (GANs) — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file includes ALL GAN-related papers from your list — foundational GANs, conditional GANs, representation learning GANs, image-to-image translation, large-scale GANs, StyleGAN family, and theoretical GAN papers.

---

## 🔥 Foundations of GANs

- [ ] **Generative Adversarial Networks (GAN)**  
      Goodfellow et al., 2014  
      https://arxiv.org/abs/1406.2661

- [ ] **Conditional GANs (cGAN)**  
      Mirza & Osindero, 2014  
      https://arxiv.org/abs/1411.1784

---

## 🧱 Convolutional & Representation Learning GANs

- [ ] **DCGAN — Deep Convolutional GANs**  
      Radford et al., 2016  
      https://arxiv.org/abs/1511.06434

- [ ] **InfoGAN — Information Maximizing GANs**  
      Chen et al., 2016  
      https://arxiv.org/abs/1606.03657

---

## 🎨 Image-to-Image Translation

- [ ] **Pix2Pix — Image-to-Image Translation with cGANs**  
      Isola et al., 2017  
      https://arxiv.org/abs/1611.07004

- [ ] **CycleGAN — Unpaired Image-to-Image Translation**  
      Zhu et al., 2017  
      https://arxiv.org/abs/1703.10593

- [ ] **StarGAN — Multi-Domain Image-to-Image Translation**  
      Choi et al., 2018  
      https://arxiv.org/abs/1711.09020

---

## 📈 Stability & Regularization in GAN Training

- [ ] **Spectral Normalization for GANs**  
      Miyato et al., 2018  
      https://arxiv.org/abs/1802.05957

- [ ] **TTUR — Two Time-Scale Update Rule for GAN Convergence**  
      Heusel et al., 2017  
      https://arxiv.org/abs/1706.08500

- [ ] **Projection Discriminator (for Conditional GANs)**  
      Miyato & Koyama, 2018  
      https://arxiv.org/abs/1802.05637

---

## 🧬 Large-Scale & High-Fidelity GANs

- [ ] **Progressive Growing of GANs**  
      Karras et al., 2018  
      https://arxiv.org/abs/1710.10196

- [ ] **BigGAN — Large Scale GAN Training**  
      Brock et al., 2019  
      https://arxiv.org/abs/1809.11096

---

## 👑 StyleGAN Family

- [ ] **StyleGAN — A Style-Based Generator Architecture**  
      Karras et al., 2019  
      https://arxiv.org/abs/1812.04948

- [ ] **StyleGAN2 — Improved Techniques for Image Quality**  
      Karras et al., 2020  
      https://arxiv.org/abs/1912.04958

- [ ] **StyleGAN3 — Alias-Free GANs**  
      Karras et al., 2021  
      https://arxiv.org/abs/2106.12423

---

## ⭐ Summary

This file includes **every GAN paper** you provided:

✔ GAN  
✔ Conditional GAN  
✔ DCGAN  
✔ InfoGAN  
✔ Pix2Pix  
✔ CycleGAN  
✔ StarGAN  
✔ Progressive GAN  
✔ Spectral Norm  
✔ TTUR  
✔ Projection Discriminator  
✔ BigGAN  
✔ StyleGAN  
✔ StyleGAN2  
✔ StyleGAN3  

Nothing missing.

# Large Language Models (LLMs) — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file includes ALL LLM papers you listed — GPT family, PaLM, Chinchilla, LLaMA, UL2, FLAN, etc.

---

## 🧠 Autoregressive Transformer Language Models

- [ ] **GPT-3 — Language Models Are Few-Shot Learners**  
      Brown et al., 2020  
      https://arxiv.org/abs/2005.14165

- [ ] **Jurassic-1 — At-Scale Autoregressive Language Models**  
      Dai et al., 2022  
      https://arxiv.org/abs/2107.02053 (closest public version; original model paper partially released)

- [ ] **Megatron-Turing NLG (530B)**  
      Smith et al., 2022  
      https://arxiv.org/abs/2201.11990

- [ ] **GLM-130B — General Language Model**  
      Zeng et al., 2022  
      https://arxiv.org/abs/2210.02414

---

## 🏗️ Scaling Laws & Compute-Optimal Training

- [ ] **Chinchilla — Training Compute-Optimal Large Language Models**  
      Hoffmann et al., 2022  
      https://arxiv.org/abs/2203.15556

---

## 🧬 Open-Source Foundation Models

- [ ] **LLaMA — Open and Efficient LLMs**  
      Touvron et al., 2023  
      https://arxiv.org/abs/2302.13971

---

## 🧩 Instruction Tuning, Task Generalization & Mixture Objectives

- [ ] **FLAN — Fine-Tuned Language Models Are Zero-Shot Learners**  
      Wei et al., 2021  
      https://arxiv.org/abs/2109.01652

- [ ] **UL2 — Unifying Language Learning Paradigms**  
      Tay et al., 2022  
      https://arxiv.org/abs/2205.05131

---

## 🌍 Massive Multilingual & Pathways-Based Models

- [ ] **PaLM — Scaling Language Modeling with Pathways (540B)**  
      Chowdhery et al., 2022  
      https://arxiv.org/abs/2204.02311

---

## 🧪 Domain-Specialized LLMs

- [ ] **Galactica — Large Language Model for Science**  
      Taylor et al., 2022  
      https://arxiv.org/abs/2211.09085

---

## 🤖 Open Community Models

- [ ] **StableLM — Stability AI Open LLMs**  
      Stability AI, 2023  
      https://github.com/Stability-AI/StableLM

---

## ⭐ Summary

This file contains **all LLM papers you listed**, including:

✔ GPT-3  
✔ PaLM  
✔ Chinchilla  
✔ LLaMA  
✔ GLM-130B  
✔ Jurassic-1  
✔ MT-NLG (Megatron-Turing)  
✔ FLAN  
✔ UL2  
✔ Galactica  
✔ StableLM  

Nothing missing.

# Vision–Language Models & Embeddings — Full Implementation Checklist  
(All Papers You Provided, With Correct Links)

This file includes ALL vision–language papers from your list — CLIP to ALIGN to ViLBERT.

---

## 🔗 Contrastive Image–Text Representation Learning

- [ ] **CLIP — Learning Transferable Visual Models from Natural Language Supervision**  
      Radford et al., 2021  
      https://arxiv.org/abs/2103.00020

- [ ] **ALIGN — Scaling Up Visual and Language Representation Learning**  
      Jia et al., 2021  
      https://arxiv.org/abs/2102.05918

---

## 🧠 Multimodal Transformer Architectures

- [ ] **ViLBERT — Pretraining Task-Agnostic Vision-and-Language Representations**  
      Lu et al., 2019  
      https://arxiv.org/abs/1908.02265

---

## ⭐ Summary

This file contains all the Vision–Language papers you listed:

✔ CLIP  
✔ ALIGN  
✔ ViLBERT  

Nothing missing.

