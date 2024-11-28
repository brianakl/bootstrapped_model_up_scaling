# Gradual Smart Scaling for Accelerated LLM Pre-training with small model initalization 


## Abstract
The quadratic complexity of self-attention[vaswani] has led to the development of methods that use smaller models as initializations for training larger Large Language Models (LLMs)[1].

While these methods typically require doubling the model size, our research demonstrates that this is not strictly necessary. 

We propose a fractional scale-up approach that provides more robust and stable model performance. 

Recent breakthroughs in normalization techniques have shown potential to drastically reduce pre-training time, achieving 4-20x convergence speed-up. 

Building on these advancements, our study introduces a novel method for incrementally scaling up models while preserving performance. 

Our findings indicate that training a model to maximum performance is not a prerequisite for successful scale-up. 

Instead, we present an incremental scaling process that maintains model efficacy throughout expansion. 

This approach offers a more efficient and flexible alternative to existing scaling methods, potentially revolutionizing the development of larger, more capable LLMs.


## Introduction

Modern LLM architectures are descendants of the original transformer architecture devised under Vaswani et al. [Vaswani et. al.].

State of the art models currently are based on a decoder framework and trained on next token prediction [gpt3, llama3].

These models also involve lengthy pre-training stages to ensure the larger models have better initalizations [gpt3, llama3].

The general consensus is that a model should be trained on 20 tokens for every parameter in it [scaling laws?, or chinchilla].

Additionally, models are being trainined on increasingly longer context lengths [cite], even though this requires increasingly higher quality data to fill the context length adaquently [cite].


There has been evidence that this is rather inefficient as models tend to performe significantly worse in testing on longer context lengths, suggesting that the quality of long context data in the training corpus is not adaquet [critical context length].

There has also been research done on efficient training, suggesting that some hyperparameters that greatly increase the complexity of training, such as context length, need not be so large [training BERT on an academic budget].

Additionally, new architectures can be used to provide faster convergence [roformer, linformer, performer, nGPT].


This leads us into our problem.

Current Transformer[vaswani] models of various sizes are all being trained on the same problem.

At a low level we can see the decoder architecture is training models to accurately predict the next token in a sequence [gpt2 paper].

We can philosophize what is actually required for next token prediction and whether or not intelligence is simply a computation. 

But, the main point to bring up in this is that models of all sizes are learning low dimensional projections of the same problem space.

This is evident in their sharing of spectral decomposition [linformer] and the rapid training speed up in the normalized GPT (nGPT) architecture that shows Transformers perform representation learning on the hypersphere.


Additionally, it has been shown that models can be simply reduced to a lower dimensional model with fewer parameters and acheive similar performance using simple dimensionality reductions techniques [find citation].

Or, when you decrease the number of parameters of a model you can only get as good or worse performance.

But, as has been repeatedly shown, the more parameters a model has the better it performs. 

A reason for this might have to do with the lottery ticket hypothesis, where larger models are more likely to have stronger and more robust subnetworks with better initalizations.

This makes removing any parameters of a larger model neccesarily make it perform worse. 

This has also been demonstrated with the results of various transformer architectures that were developed following Vaswani et. al. [linformer, Performer, Reformer].


The current LLM training loop requires a lengthy warmup stage where up to 10% of the total training tokens are used to warm up the model[citation needed] to ensure a better initalization. 

Additionally, these models require batch sizes upwards of 1 million tokens, and context lengths of 4096 tokens or more [gpt3].

This is radically inefficent as the training context length often ends up being less than the effective context lenght in testing [critical batch size].


Most methods that achieve better model performance involve modifying the architecture to let the model learn faster [Roformer, nGPT]. 

These methods do not involve changing the model size but rather finding a way to force the model to learn faster. 






## Methodology
lorem

## Experiments
lorem

## Conclusion
lorem

## References
[1] Scaling Smart
[2] nGPT

## Appendix

