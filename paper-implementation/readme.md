
1

# Automatic Zoom

## Natural Language Processing (NLP)

| Paper Name | Link | Description |
| --- | --- | --- |
| “Efficient Estimation of Word Representations in Vector Space” (Tomas Mikolov et al., 2013) | [Link](https://arxiv.org/abs/1301.3781) | Introduces word2vec (skip-gram and CBOW) to learn continuous word embeddings. Word2vec efficiently produces high-quality word vectors that capture precise syntactic and semantic relationships . (Code: Mikolov’s word2vec). |
| “Sequence to Sequence Learning with Neural Networks” (Ilya Sutskever et al., 2014) | [Link](https://arxiv.org/abs/1409.3215) | First to show an end-to-end LSTM encoder-decoder that maps input sequences to output sequences. Achieved strong machine translation results (34.8 BLEU on English–French) by training two LSTMs jointly . (Code: e.g. TensorFlow seq2seq example). |
| “Neural Machine Translation by Jointly Learning to Align and Translate” (Dzmitry Bahdanau et al., 2015) | [Link](https://arxiv.org/abs/1409.0473) | Introduces the attention mechanism for NMT. By allowing the decoder to “attend” to relevant encoder states dynamically, it overcomes the fixed-size context bottleneck. The model achieved competitive translation performance and produces interpretable soft-alignments . (Code: Bahdanau’s NMT code). |
| “Attention Is All You Need” (Vaswani et al., 2017) | [Link](https://arxiv.org/abs/1706.03762) | Proposes the Transformer architecture using only self-attention (no recurrent or convolutional layers). Transformers parallelize better and learn dependencies more directly. Vaswani et al. demonstrate state-of-the-art machine translation and faster training (BLEU 28.4 on EN–DE) with fully attention-based models . (Code: Tensor2Tensor / OpenNMT). |
| “Deep Contextualized Word Representations” (ELMo) (Peters et al., 2018) | [Link](https://arxiv.org/abs/1802.05365) | Introduces ELMo, contextual word embeddings from a biLSTM language model. ELMo captures word meaning from context (syntax/semantics), improving diverse tasks (QA, NLI, sentiment) by large margins . (Code: AllenNLP ELMo). |
| “Universal Language Model Fine-Tuning for Text Classification” (ULMFiT) (Howard & Ruder, 2018) | [Link](https://arxiv.org/abs/1801.06146) | Demonstrates that transfer learning (pre-train on large corpus, then fine-tune on task) works for NLP. ULMFiT applies discriminative fine-tuning and slanted triangular learning rates to LSTM language models. It obtains huge error reductions (18–24%) on multiple text classification benchmarks . (Code: stperrakis/ulmfit). |
| “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” (Devlin et al., 2018) | [Link](https://arxiv.org/abs/1810.04805) | Introduces BERT, a deep Transformer model pre-trained on masked language modeling and next-sentence prediction. BERT’s bidirectional context and fine-tuning scheme yield state-of-the-art results on many benchmarks (GLUE, SQuAD) with minimal architecture changes . (Code: google-research/bert ). |
| “Improving Language Understanding by Generative Pre-Training” (GPT) (Radford et al., 2018) | [Link](https://arxiv.org/abs/1805.00932) | Presents GPT (OpenAI’s first large Transformer LM). Shows that unsupervised pre-training on diverse text, followed by discriminative fine-tuning, yields strong gains on multiple NLP tasks. GPT (117M parameters) outperforms many baselines on 9 of 12 NLP benchmarks without heavy task-specific architecture . (Code: openai/finetune-transformer-lm). |
| “Language Models are Unsupervised Multitask Learners” (GPT-2) (Radford et al., 2019) | [Link](https://arxiv.org/abs/1810.04805) | Introduces GPT-2, a 1.5B-parameter Transformer. Without fine-tuning, GPT-2 performs many tasks (QA, translation, summarization) in a zero/few-shot manner by conditioning on textual prompts. The model matches or exceeds SOTA on 7/8 language benchmarks and can generate highly coherent paragraphs . (Code: openai/gpt-2). |

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11

| “RoBERTa: A Robustly Optimized BERT Pretraining Approach” (Liu et al., 2019) | [Link](https://arxiv.org/abs/1907.11692) | Shows BERT can be improved by training longer/bigger on more data with hyperparameter tweaks. RoBERTa removes the next-sentence objective, uses dynamic masking, larger batches, and larger corpora. The result matches or exceeds all models since BERT: RoBERTa’s best model sets SOTA on GLUE, RACE, and SQuAD . (Code: facebookresearch/fairseq includes RoBERTa). |
| “XLNet: Generalized Autoregressive Pretraining for Language Understanding” (Yang et al., 2019) | [Link](https://arxiv.org/abs/1906.08237) | Introduces XLNet, which combines autoregressive and autoencoding: it maximizes likelihood over permutations of input tokens (via Transformer-XL backbone). XLNet captures bidirectional context without masking. It significantly outperforms BERT on 20+ tasks, notably QA and NLI . (Code: CMU-Kit/XLNet). |
| “ALBERT: A Lite BERT for Self-supervised Learning of Language Representations” (Lan et al., 2019) | [Link](https://arxiv.org/abs/1909.11942) | Proposes ALBERT, which shares parameters across layers and factorizes embeddings to reduce model size. Also introduces a sentence-order prediction loss. Despite far fewer parameters, ALBERT matches or exceeds BERT-large: 33× fewer parameters achieve new SOTA on GLUE/RACE/SQuAD . (Code: google-research/albert). |
| “BART: Denoising Sequence-to-Sequence Pre-training for NLG, Translation, and Comprehension” (Lewis et al., 2019) | [Link](https://arxiv.org/abs/1910.13461) | Introduces BART, a seq2seq Transformer pretrained as a denoising autoencoder (corrupt text with noise, reconstruct). BART combines a bidirectional encoder (like BERT) and autoregressive decoder (like GPT). When fine-tuned, it matches RoBERTa on language understanding and achieves new SOTA on generation tasks (summarization, dialogue, QA) . (Code: facebookresearch/fairseq includes BART). |
| “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” (T5) (Raffel et al., 2020) | [Link](https://arxiv.org/abs/1910.10683) | Casts every NLP task as text generation (“text-to-text”) and scales a Transformer up to 11B parameters. Trained on massive cleaned web crawl (C4), T5 achieves SOTA results on a wide range of tasks (summarization, QA, translation, etc.) by simply framing tasks as text transformation . (Code: google-research/text-to-text-transfer-transformer). |
| “ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators” (Clark et al., 2020) | [Link](https://arxiv.org/abs/2003.10555) | Proposes ELECTRA, a more sample-efficient pretraining: a small generator replaces tokens (like masked LM), and a larger discriminator must detect replaced tokens. This uses all tokens for learning. ELECTRA trains faster: small ELECTRA models outperform much larger BERT/GPT with the same compute . (Code: google-research/electra). |
| “Dense Passage Retrieval for Open-Domain Question Answering” (DPR) (Karpukhin et al., 2020) | [Link](https://arxiv.org/abs/2004.04906) | Proposes DPR, a dual-encoder dense retrieval approach for open-domain QA. Questions and passages are embedded with BERT encoders, and nearest-neighbor retrieval finds relevant docs. DPR outperforms BM25 by 9–19% in top-20 recall, enabling state-of-the-art QA when combined with a reader model . (Code: facebookresearch/DPR). |
| “Retrieval-Augmented Generation (RAG)” (Lewis et al., 2020) | [Link](https://arxiv.org/abs/2005.11401) | Combines generative seq2seq models with a non-parametric retriever over a knowledge corpus. RAG fetches relevant documents for each query and conditions generation on them. Two variants (RAG-Sequence and RAG-Token) achieve SOTA on 3 open-domain QA tasks, surpassing both purely neural models and pipeline methods. RAG produces more factual answers by referencing retrieved passages . (Code: huggingface/transformers RAG examples). |
| “SpanBERT: Improving Pre-training by Representing and Predicting Spans” (Joshi et al., 2020) | [Link](https://arxiv.org/abs/1907.10529) | Extends BERT by masking contiguous spans (not just single tokens) and training to predict entire spans. SpanBERT also trains span-boundary representations. This yields large gains on span-based tasks: SOTA on QA (SQuAD F1 94.6) and coreference (OntoNotes F1 79.6) . (Code: facebookresearch/SpanBERT). |

- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26

2

# Automatic Zoom

## Curated Advanced Papers — Deep Learning | Vision–Language & Embeddings | ML/AI, Generative & Agentic AI

Prepared: 2025-11-15 17:07 UTC

### Deep Learning — Core theory & applied milestones (≈20 papers)

| Paper Name | Link | Description |
| --- | --- | --- |
| Efficient BackProp — Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller (1998) | [Link](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) | Classic tutorial covering optimization, nonlinearities, initialization — essential background on training deep nets. |
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Nitish Srivastava et al. (2014) | [Link](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) | Introduces dropout regularization, a simple stochastic neuron dropping method that reduces overfitting and co-adaptation. Code: https://github.com/nyu-dl/dl4cv-2017-assignments/tree/master/assignment2/dropout |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift — S. Ioffe, C. Szegedy (2015) | [Link](https://arxiv.org/abs/1502.03167) | BatchNorm stabilizes training, enables larger learning rates, and reduces sensitivity to initialization; widely used. Code: https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet |
| Adam: A Method for Stochastic Optimization — Diederik P. Kingma, Jimmy Ba (2015) | [Link](https://arxiv.org/abs/1412.6980) | Popular adaptive optimizer combining momentum and adaptive learning rates; baseline for many deep learning tasks. Code: https://github.com/adam-p/adam |
| Understanding the difficulty of training deep feedforward neural networks (Xavier init) — Xavier Glorot, Yoshua Bengio (2010) | [Link](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) | Introduces Xavier/Glorot initialization to keep signal variance stable across layers. |
| Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU) — A. Krizhevsky, I. Sutskever, G. Hinton (2010) | [Link](https://papers.nips.cc/paper/2010/file/1fb3ac3a8b0d0c3b2c3c4b4b0f1b8b22-Paper.pdf) | Popularizes ReLU nonlinearities for faster convergence in deep nets. |
| ResNet: Deep Residual Learning for Image Recognition — Kaiming He et al. (2015) | [Link](https://arxiv.org/abs/1512.03385) | Introduces residual connections allowing very deep networks to be trained (ResNet family). Code: https://github.com/KaimingHe/deep-residual-networks |
| Network in Network — Min Lin, Q. Chen, S. Yan (2013) | [Link](https://arxiv.org/abs/1312.4400) | Micro-architectural idea (MLP conv layers) that influenced modern conv designs. |
| Squeeze-and-Excitation Networks — Jie Hu et al. (2017) | [Link](https://arxiv.org/abs/1709.01507) | Channel-wise attention block that boosts performance with small cost. Code: https://github.com/hujie-frank/SENet |
| Attention Is All You Need — Vaswani et al. (2017) | [Link](https://arxiv.org/abs/1706.03762) | Introduces Transformers — key architecture across modalities. Code: https://github.com/tensorflow/tensor2tensor |
| Vision Transformer (ViT) — Dosovitskiy et al. (2020) | [Link](https://arxiv.org/abs/2010.11929) | Shows pure Transformers can succeed on image tasks when scaled and pre-trained. Code: https://github.com/google-research/vision_transformer |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks — Mingxing Tan & Quoc V. Le (2019) | [Link](https://arxiv.org/abs/1905.11946) | Proposes compound scaling to balance depth/width/resolution for efficient accuracy. Code: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet |
| Large Batch Training of Convolutional Networks — P. Goyal et al. (2017) | [Link](https://arxiv.org/abs/1706.02677) | Techniques for training with very large batch sizes (linear scaling rule, warmup). |
| Layer Normalization — Jimmy Lei Ba et al. (2016) | [Link](https://arxiv.org/abs/1607.06450) | Normalization method applied across features, important for RNNs and Transformers. |
| Weight Decay, Regularization and Generalization in Deep Nets — Various key references (1992) | [Link](https://link.springer.com/chapter/10.1007/3-540-55719-9_3) | Classic theory on regularization with implications for modern deep learning. |
| SimCLR: A Simple Framework for Contrastive Learning of Visual Representations — Ting Chen et al. (2020) | [Link](https://arxiv.org/abs/2002.05709) | SimCLR shows contrastive learning can produce strong representations without labels. Code: https://github.com/google-research/simclr |
| BYOL: Bootstrap Your Own Latent — Jean-Bastien Grill et al. (2020) | [Link](https://arxiv.org/abs/2006.07733) | Self-supervised method that avoids negative samples; strong representation learning. Code: https://github.com/deepmind/deepmind-research/tree/master/byol |
| DINO: Self-Distillation with No Labels — Mathilde Caron et al. (2021) | [Link](https://arxiv.org/abs/2104.14294) | Shows ViTs can learn good features without supervision using self-distillation. Code: https://github.com/facebookresearch/dino |
| Stochastic Depth and DropPath — G. Huang et al. (2016) | [Link](https://arxiv.org/abs/1603.09382) | Training-time layer dropping to regularize very deep networks. |
| Understanding Generalization in Deep Learning — Zhang et al., Neyshabur et al., etc. (2016-2019) | [Link](https://arxiv.org/abs/1611.03530) | Insights into optimization and generalization puzzles of deep nets. |

### Vision–Language Models & Embeddings (≈20 papers)

| Paper Name | Link | Description |
| --- | --- | --- |
| CLIP: Learning Transferable Visual Models From Natural Language Supervision — Radford et al. (OpenAI) (2021) | [Link](https://arxiv.org/abs/2103.00020) | Contrastive pretraining linking images and text; strong zero-shot transfer across vision tasks. Code: https://github.com/openai/CLIP |
| ALIGN: Scaling Up Visual and Language Representation Learning — Jia et al. (Google) (2021) | [Link](https://arxiv.org/abs/2102.05918) | Large-scale image-text contrastive learning using noisy web alt-text; CLIP-like results at scale. Code: https://github.com/google-research/vision_transformer |
| ViLBERT: Pretraining Task-Agnostic V+L Representations — Lu et al. (2019) | [Link](https://arxiv.org/abs/1908.02265) | Two-stream model processing image regions and text with co-attention; strong VQA and captioning. Code: https://github.com/facebookresearch/vilbert-multi-task |
2

1

Automatic Zoom
Curated Advanced Papers — Deep Learning |
Vision–Language & Embeddings | ML/AI, Generative &
Agentic AI

Prepared: 2025-11-15 17:07 UTC

Deep Learning — Core theory & applied milestones (≈20 papers)

Efficient BackProp — Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller (1998)

Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

Classic tutorial covering optimization, nonlinearities, initialization — essential background on training deep nets.

Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Nitish Srivastava et al. (2014)

Paper: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf | Code:
https://github.com/nyu-dl/dl4cv-2017-assignments/tree/master/assignment2/dropout

Introduces dropout regularization, a simple stochastic neuron dropping method that reduces overfitting and
co-adaptation.

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift — S.
Ioffe, C. Szegedy (2015)

Paper: https://arxiv.org/abs/1502.03167 | Code:
https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet

BatchNorm stabilizes training, enables larger learning rates, and reduces sensitivity to initialization; widely used.

Adam: A Method for Stochastic Optimization — Diederik P. Kingma, Jimmy Ba (2015)

Paper: https://arxiv.org/abs/1412.6980 | Code: https://github.com/adam-p/adam

Popular adaptive optimizer combining momentum and adaptive learning rates; baseline for many deep learning tasks.

Understanding the difficulty of training deep feedforward neural networks (Xavier init) — Xavier
Glorot, Yoshua Bengio (2010)

Paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

Introduces Xavier/Glorot initialization to keep signal variance stable across layers.

Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU) — A. Krizhevsky, I. Sutskever,
G. Hinton (2010)

Paper: https://papers.nips.cc/paper/2010/file/1fb3ac3a8b0d0c3b2c3c4b4b0f1b8b22-Paper.pdf

Popularizes ReLU nonlinearities for faster convergence in deep nets.

ResNet: Deep Residual Learning for Image Recognition — Kaiming He et al. (2015)

Paper: https://arxiv.org/abs/1512.03385 | Code: https://github.com/KaimingHe/deep-residual-networks

Introduces residual connections allowing very deep networks to be trained (ResNet family).

Network in Network — Min Lin, Q. Chen, S. Yan (2013)

Paper: https://arxiv.org/abs/1312.4400

Micro-architectural idea (MLP conv layers) that influenced modern conv designs.

Squeeze-and-Excitation Networks — Jie Hu et al. (2017)

Paper: https://arxiv.org/abs/1709.01507 | Code: https://github.com/hujie-frank/SENet

Channel-wise attention block that boosts performance with small cost.

Attention Is All You Need — Vaswani et al. (2017)

Paper: https://arxiv.org/abs/1706.03762 | Code: https://github.com/tensorflow/tensor2tensor

Introduces Transformers — key architecture across modalities.
Vision Transformer (ViT) — Dosovitskiy et al. (2020)

Paper: https://arxiv.org/abs/2010.11929 | Code: https://github.com/google-research/vision_transformer

Shows pure Transformers can succeed on image tasks when scaled and pre-trained.

EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks — Mingxing Tan & Quoc V.
Le (2019)

Paper: https://arxiv.org/abs/1905.11946 | Code: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

Proposes compound scaling to balance depth/width/resolution for efficient accuracy.

Large Batch Training of Convolutional Networks — P. Goyal et al. (2017)

Paper: https://arxiv.org/abs/1706.02677

Techniques for training with very large batch sizes (linear scaling rule, warmup).

Layer Normalization — Jimmy Lei Ba et al. (2016)

Paper: https://arxiv.org/abs/1607.06450

Normalization method applied across features, important for RNNs and Transformers.

Weight Decay, Regularization and Generalization in Deep Nets — Various key references (1992)

Paper: https://link.springer.com/chapter/10.1007/3-540-55719-9_3

Classic theory on regularization with implications for modern deep learning.

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations — Ting Chen et al.
(2020)

Paper: https://arxiv.org/abs/2002.05709 | Code: https://github.com/google-research/simclr

SimCLR shows contrastive learning can produce strong representations without labels.

BYOL: Bootstrap Your Own Latent — Jean-Bastien Grill et al. (2020)

Paper: https://arxiv.org/abs/2006.07733 | Code: https://github.com/deepmind/deepmind-research/tree/master/byol

Self-supervised method that avoids negative samples; strong representation learning.

DINO: Self-Distillation with No Labels — Mathilde Caron et al. (2021)

Paper: https://arxiv.org/abs/2104.14294 | Code: https://github.com/facebookresearch/dino

Shows ViTs can learn good features without supervision using self-distillation.

Stochastic Depth and DropPath — G. Huang et al. (2016)

Paper: https://arxiv.org/abs/1603.09382

Training-time layer dropping to regularize very deep networks.

Understanding Generalization in Deep Learning — Zhang et al., Neyshabur et al., etc. (2016-2019)

Paper: https://arxiv.org/abs/1611.03530

Insights into optimization and generalization puzzles of deep nets.

Vision–Language Models & Embeddings (≈20 papers)

CLIP: Learning Transferable Visual Models From Natural Language Supervision — Radford et al.
(OpenAI) (2021)

Paper: https://arxiv.org/abs/2103.00020 | Code: https://github.com/openai/CLIP

Contrastive pretraining linking images and text; strong zero-shot transfer across vision tasks.

ALIGN: Scaling Up Visual and Language Representation Learning — Jia et al. (Google) (2021)

Paper: https://arxiv.org/abs/2102.05918 | Code: https://github.com/google-research/vision_transformer

Large-scale image-text contrastive learning using noisy web alt-text; CLIP-like results at scale.

ViLBERT: Pretraining Task-Agnostic V+L Representations — Lu et al. (2019)

Paper: https://arxiv.org/abs/1908.02265 | Code: https://github.com/facebookresearch/vilbert-multi-task

Two-stream model processing image regions and text with co-attention; strong VQA and captioning.
