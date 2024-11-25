******************** WORK IN PROGRESS (Draft 0.1.0) *********************


# Bootstrapped Up Scaling (BUS) / The Student Becomes the Teacher
## Abstract

The objective of this research is to explore whether knowledge learned by a smaller Transformer model can be transferred to a larger one. This could significantly improve the efficiency of training Large Language Models (LLMs) by reducing the computational cost associated with pretraining, which often involves extensive matrix multiplications. By investigating the possibility of pretraining a smaller model and then scaling it up to a larger architecture, bootstrapped  up scaling (BUS), we aim to identify potential time- and cost-saving strategies for LLM training.

## Introduction
The pretraining of LLMs has become increasingly computationally expensive, requiring vast amounts of memory and processing power. A key factor contributing to this challenge is the sheer size of the models themselves, which necessitate extensive matrix multiplications during training. Other transformer architectures that aim to improve model performance tend to sacrifice performance for speed [linformer, performer, reformer]. The current literature suggests that larger models do perform better due to the ability to generate good subnetworks (lottery ticket hypothesis) []. This paper explores a potential solution to mitigate these costs by investigating whether knowledge learned by smaller Transformer models can be transferred to larger ones, enabling more efficient training of LLMs without the cost of performance from training a smaller model.

## Background

The motivating idea behind this is the Singular Value Decomposition (SVD)[]. SVD enables the condensation of large matrices into smaller ones through linear dimensionality reduction. This raises an intriguing question: if we can shrink a large matrix to a smaller one, can we also scale up from small to large? While perfect reconstruction is unlikely, might it be possible to transfer some information from the smaller matrix to its larger counterpart?

This idea is inspired by the observation that both smaller and larger models are tackling the same problem, suggesting they share common characteristics. Notably, they exhibit similar approximate eigenvalues [insert spectral analysis]. As smaller matrices struggle to capture optimal solutions due to limited dimensionality, their training can drive them closer to the eigenvalues of the larger matrix. Which then 'guides' the larger model in the direction of the optimal solution. This implies that the smaller model learns a lower-dimensional projection of the full solution.

The existing literature on the subject currently revolves around making the attention mechanism faster via the addition of some linear transformations to take advantage of the fact that Self-Attention is low rank [Wang et al.]. This however, is a change to the original transformer architecture, reducing the dimensionality, and ergo reducing performance.

The next challenge lies in developing a method to scale up this lower-dimensional solution to a larger matrix, effectively "inflating" the smaller model's capabilities.

## Methodology
As a proof of concept, a relatively simple task was chosen such that it can be trained locally and quickly. The model used was also small by design to provide give an easy way to compare learning rates.

Additionally, the arcitecture was tested on a character level decoder LLM, before expanding to a full-sized LLM 

### Test
To test the BUS capabilty of transformers a simple task was used to test transformer learning. This task is one that a transformer would excel at in comparison to other machine learning models/architectures.

### The Task

The objective of this task is to develop a predictive model that, given a string of characters, outputs the frequency count of each character at its corresponding position. The goal is to predict the number of times a particular character has occurred before, with a maximum count of 2. This makes it a 3 class classification problem.

This task serves as an ideal benchmark for evaluating the performance of various models, particularly transformer-based architectures. It can be learned with a single-headed transformer layer without requiring multiple layers or large neural networks. Consequently, the performance of the model is directly proportional to that of the underlying transformer architecture.

The self-attention mechanism inherent in transformers enables them to effectively "look back" at the input sequence, making this task particularly well-suited for evaluation purposes.

**Example Inputs**

Below are two example inputs:

```
we propose a new simp
000000110110202021002
men love the downfall
000000011002201010001
```
In each example, the output should indicate the frequency count of each character at its corresponding position.

**Training Set Specifications**

Each string in the training set will consist of 20 characters. The vocabulary is limited to English letters and spaces, with numbers represented as individual digits (e.g., "10" becomes "one zero").


### The dataset
The dataset was taken from text8.



### Model Architecture

| d_model | batch size | Max Learning Rate | Number of Heads|
|:--------------------:|:----------------:|:--------------------------------:|:--------------------:|
| 64 | 81.4% | 86.8% | +5.9% |
| 128 | 88.0% | 93.6% | +5.6% |
| 256 | 91.2% | 96.8% | +5.6% |
| 512 | 92.3% | 97.7% | +5.4% |
| 1024| 93.7% | 99.1% | +5.4% |



### Steps 
- attempted doubling every col and row
    - unsuccessful attempt, no learning improvement
- adding one hot vectors to each matrix to expand it that way
    - this works, knowlede from a 
    - test protocol:
        - Train a transformer for 20 epochs at half the size n/2
        - expand the transformer to full size by simply adding it to the n sized identity
        - train for an additional 30 epochs
        - train a seperate model at full size n for 50 epochs
        - compare average dev set accuracy
        - this test was performed 25 times and averaged     
        - test still needs to be performed more to obtain a statistically valid result
    - results:
        - with smaller models that acheive worse performance, there seems to be little to no improvement, this is with matrix sizes of <64
        - with larger models, the improvement is larger

![Matrix example](images/BUS_ex.png)

## Results

#### Letter Counting Test

| Model size (d_model) | Dev set accuracy | Dev accuracy (transfer learning) | Transfer Improvement |
|:--------------------:|:----------------:|:--------------------------------:|:--------------------:|
| 64 | 81.4% | 86.8% | +5.9% |
| 128 | 88.0% | 93.6% | +5.6% |
| 256 | 91.2% | 96.8% | +5.6% |
| 512 | 92.3% | 97.7% | +5.4% |
| 1024| 93.7% | 99.1% | +5.4% |

![Accuracy across epochs (model size 1024)](images/acc_model_1024.png)
![Loss across epochs (model size 1024)](images/loss_model_1024.png)

![Accuracy across epochs (model size 512)](images/acc_model_512.png)
![Loss across epochs (model size 512)](images/loss_model_512.png)

![Accuracy across epochs (model size 256)](images/acc_model_256.png)
![Loss across epochs (model size 256)](images/loss_model_256.png)

![Accuracy across epochs (model size 128)](images/acc_model_128.png)
![Loss across epochs (model size 128)](images/loss_model_128.png)


#### LLM Test

| Model size (d_model) | Dev set accuracy | Dev accuracy (transfer learning) | Transfer Improvement |
|:--------------------:|:----------------:|:--------------------------------:|:--------------------:|
| 64 | 81.4% | 86.8% | +5.9% |
| 128 | 88.0% | 93.6% | +5.6% |
| 256 | 91.2% | 96.8% | +5.6% |
| 512 | 92.3% | 97.7% | +5.4% |
| 1024| 93.7% | 99.1% | +5.4% |

### Expansion to Larger Task

### Statistical Analysis

## Analysis
- standard deviation of the non-boostrapped model is much higher, suggesting that it's taking much longer to converge

### Attempts to disprove
- Possibly effect of identity matrix initialization
    - using an identity initalization for the matrices performs about equal to standard torch.nn.linear init

### Optimal transfer training stopping

### Limitations

## Conclusion


## References
******************************* workspace ******************************
count = 20
- Attention is all you need, Vaswani et. al.,  	arXiv:1706.03762 [cs.CL]
- Linformer: Self-Attention with Linear Complexity, Wang et al. arXiv:2006.04768 [cs.LG]
- Preformer: Predictive Transformer with Multi-Scale Segment-wise Correlations for Long-Term Time Series Forecasting, Du et al. arXiv:2202.11356 [cs.LG]
- Reformer: The Efficient Transformer, arXiv:2001.04451 [cs.LG]
- Scaling Laws for Neural Language Models, Kaplan et al. arXiv:2001.08361 [cs.LG]
- How to train BERT with an academic budget, Izsak et al. https://aclanthology.org/2021.emnlp-main.831.pdf
- BERT: ..., Devlin et al. arXiv:1810.04805 [cs.CL]
- The Lottery Ticket hypothesis: finding sparse, trainable neural networks, Frankle et al. arXiv:1803.03635 [cs.LG]
- Rethinking attention with performers, Choromanski et al.  	arXiv:2009.14794 [cs.LG]
- Improving Language Understanding by generative pre-training, Radford et al. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al. arXiv:2104.09864 [cs.CL]
- Are Sixteen Heads Really Better than One? https://proceedings.neurips.cc/paper_files/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf
- Language Models are Few-Shot Learners. arXiv:2005.14165 [cs.CL]
- The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI]
- Training Compute-Optimal Large Language Models.  	arXiv:2203.15556 [cs.CL]
- Why does the effective context length of LLMs fall short? arXiv:2410.18745v1 [cs.CL] 24 Oct 2024
- FlashAttention-2: Faster Attention with better Parallelism with Working Partitioning  	arXiv:2307.08691 [cs.LG]
- nGPT: Normalized Transformer with Representation Learning on the Hypersphere arXiv:2410.01131 [cs.LG]
- How Does Critical Batch Size Scale in Pre-Training? arXiv:2410.21676 [cs.LG]- 
- GPT-NeoX @software{gpt-neox-library,
  title = {{GPT-NeoX: Large Scale Autoregressive Language Modeling in PyTorch}},
  url = {https://www.github.com/eleutherai/gpt-neox},
- 


## Personal Notes

- if generally a transformer learns via gradient descent, and you are training two models on the same problem, if one model is smaller than the other then isn't the smaller model learning a projection of the problem space in a lower dimension?
- if that is the case then the there could be some overlap in the solutions that can be found as you scale the model up, since it'll get a higher resolution image of the terrain of the problem space
- for example, if you take a paraboloid and project it onto a 2d plane, you can get different perspectives of the problem, some useful and some not so useful, but of the useful ones, the low dimensional solution does share some information with the full dimensional solution
- what does it mean for two matrices to have the same singular values?
- Since these matrices all seem to converge to the same singular values there must be something that they are all converging to 
- If they share singular values does that mean that we can construct a new larger matrix based on the singular values of the smaller matrix that will be closer to the solution for the larger matrix?




<!-- - TODO: -->
<!--     - compare what kind of transfer learning is best -->
<!--     - compare transfer training ratio -->
<!--     - compare ammount of transfer training -->
<!--     - plot the training rates (dev set accuracy) -->
<!--         - compare epoch to epoch  -->
<!--         - compare epoch to epoch for same model size -->
<!--         - the transfer learning should show a much sharper learning rate -->
<!--         - also compare training loss? why not  -->
<!--    - See if there is an improvement on the efficient compute boundary with this approach? -->





