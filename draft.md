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

This approach also opens the door for future work that could enable efficient scaling of both model width and depth.


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


The Roformer architecture provides a more effective way to insert postional information into the self-attention process.

The original method using sinusodial embeddings at the model's input proves to be ineffective since downstream layers, the positional information gets lost [find citation].

With the Roformer we directly insert the positional information into every self attention step.

This has allowed for faster convergence across training [Roformer].



The recent nGPT (normalized GPT) architecture has also led to a significant speed up in training speed.

In addition, there is also a normalization step after each batch step, details of the implementation can be found in Loshchilov et al. [nGPT].


table:
normal transformer
    - x_a = ATTN(LayerNorm(x))
    - x = x + x_a
    - x_b = MLP(LayerNorm(x))
    - x_b = x + x_b
    - x = LayerNorm(x)

nGPT:
    - x_a = Norm(Attn(x))
    - x = Norm(x + a(x_a - x))
    - x_b = Norm(MLP(x))
    - x = Norm(x + b(x_b - x))

A method of scaling samller models to use as initalizations for larger models has been developed concurrently with our research. 

Their HyperCloning scaling involves doubling the size of all matrices with clones of the original matrix taking up all four quadrants of the new matrix.

They were able to demonstrate substantial downstream model improvements after performing Hyperscaling.


This lends credence to our formulation which uses their findings that the best initalization is a simple symmetric expansion of the model dimensions.

Additionally, they demonstrated that there is a limit of how much information can be transfered from the samller model to the larger one, indicating that there is an optimal amount to pretrain the smaller model for hyperscaling.

Where their experiment was lacking was in the models inability to scale continuously.

This smoother scaling technique allows the model effectively ride the most learning efficient part of training, while also taking advantage of having a larger model then learn more complex patterns in the language.



## Methodology

The main idea this scaling procedure is to train for the 20% of training tokens that account for 80% of the performance.

Since effect on model performance of each new token seen by the model decreases, this suggests that there is an optimal ammount of training to do on a smaller model to maximize amount of knowledge transfered and minimize total number of tokens seen.

A sort of pareto distribution.

This usually is about the first 20% of training tokens seen, after that the effeicency of training effect of each new token seen decreases precipteously.

Following the methodologies of Samragh et al. [scaling smart], a modified version of HyperCloning was implemented.

Namely, there are muliple times where hypercloning happens in training and the hyper cloning itself was done fractionally and not by merely doubling the matrix each time.


## Experiments
lorem

## Conclusion
lorem

## References
[1] Scaling Smart
[2] nGPT

## Appendix

