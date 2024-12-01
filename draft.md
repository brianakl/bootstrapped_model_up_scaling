# HCnGPT: HyperCloning nGPT (normalized GPT)

## Abstract
Recent advancements in language model scaling have highlighted the potential for more efficient and flexible approaches to developing larger, more capable models. 

This paper introduces HCnGPT (HyperCloning normalized GPT), a novel method that combines the efficiency gains of the normalized GPT (nGPT) architecture with a fractional scaling technique we call BUS (Bootstrapped model UpScaling). 

Our research demonstrates that 1. the nGPT architecture provides significant improvements in training efficiency 2. fractional scaling, inspired by but improving upon HyperCloning, enables more gradual and flexible model growth 3. the introduction of the \kappa parameter allows for tuning the optimal point at which to scale the model.


We present experimental results on a 64M parameter model, showcasing the potential of this approach for developing LLMs that are not only larger but also more efficient and robust. 

Our findings indicate that training a model to maximum performance is not a prerequisite for successful scale-up. 

Instead, we demonstrate an incremental scaling process that maintains model efficacy throughout expansion. 

This research opens new avenues for efficient scaling of both model width and depth, potentially revolutionizing the development of larger language models. 

By enabling more efficient use of computational resources and smoother transitions between model sizes, HCnGPT presents a promising direction for the future of LLM development.


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

Typical LLM loss graphs show that about the first 20% of tokens seen account for ~90% of the total loss improvements.

### nGPT Architecture
As mentioned before the nGPT architecture focuses on normalizing the weight matrices of the transformer architecture.

This involves removing all exisiting normalization layers in the model and replacing them with a standard vector normalization across the model dimentions.

Loshchilov et al. sugest that this supports representational learning on the hypersphere and show how the nGPT model performs better implicit gradient descent as a meta-optimiztion [cite nGPT].

Something transformers already perform but at a better level with nGPT.


The primary reason for using this architecture was for the significant efficency gains.

The nGPT architecture allows for direct training with no warm up or weight decay and reaching model capacity 4-20x faster [nGPT].

As the authors have mentioned, they were able to reach a models capacity for a given parameter count significantly faster.

To expand on this we incorporate the finding of [scaling smart] to scale the nGPT model

The standard self attention mechanism is changed as follows:

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

Although, as the authors mentioned, this adds some overhead per step that can be made more efficient with the creation of fused kernels for his operation. However, even with this overhead the nGPT architecture has a faster wall time learning rate (figure x).

### HyperCloning
The HyperCloning algorithm introduces an efficient method of model growth by allowing an effective way to expand the models width.

This provides better initalization for a larger model resulting in better downstream performance with a reduced risk of failure due to poor model initalization, loss divergence, or poor hyperparameter tuning [hyperclone].

The authors show that with HyperCloning they are able to demonstrate a 2-4x speed up of training speed and and improvement of final model performance when compared to standard random model weight initalizations.


Hypercloning works as follows:

W_L = [ W_s/2  W_s/2
        W_s/2  W_s/2]

This allows the model to perserve the performance with scaling since the output logits would be the same from one model to the next.

### BUS
Some limitations of the Hypercloning method described is that you necessarily need to double the model parameters in order to scale it.

Our findings demonstrate that hyperscaling can also be done fractionally.

This is done to allow a more gradual transition to the larger model with the idea behind it that you can use more compute resources on the larger model by scaling the model faster and more gradually.

This idea is motivated by the lottery ticket hypothesis, where the larger model will perform better the more that it is trained.


This introduces a new hyperparameter into training, let x be the predicted performance of the smaller model. 

Since we are using smaller models, we can estimate using the performance of existing models as a heuristic for what the performance of this model should get to. 

We use existing models since there is no need to waste compute on performing a predictive analysis on the intermediary model.

Let \kappa be the percent of performance that we want to extract from the smaller model.

Where the total performance is the difference between starting validation loss and final validation loss.

We can then use the following formula to determine at what validation loss we should scale.


tp * \kappa + mp = scaling target

tp = final performance - starting performance
\kappa = scaling ratio
mp = predicted final model performance


### Model Details
Due to compute limitation a very small model final model was chose of 64M parameters.

The architecture choosen was a decoder model with Rotary Positional encoding [Roformer] as this is the current State-of-the-Art architecture.

The modifications mentioned above of using the nGPT framework were also done to maximize performance.


### Benchmark Performance

As a benchmark we choose the NLI task while analyzing performance on both the SNLI and MNLI datasets. 

This task was choosen due to its simplicity and the fact that the models of this small parameter count struggle on more standard benchmarks such as Hellaswag, openbookqa, arc, and winograde[cite all datasets].


## Conclusion
Our research introduces a novel approach to scaling language models, combining the efficiency of nGPT architecture with a fractional HyperCloning technique we call BUS (Bootstrapped model UpScaling).

This Method Addresses several key challenges in the development of larger, more capable LLMs.


By leveraging the nGPT architecture, we achieve significant improvements in training efficiency, with 4-20x faster convergence compared to traditional methods. 

This allows us to capitalize on the rapid inital learning phase of model training, where approximately 15% of training tokens account of about 85% of total loss improvements.


Our fractional scaling approach, inspired by but improving upon HyperCloning, enables more gradual and flexible model growth. 

This strategy allows for, optimal utilization of compute resources, smoother transitions between model sizes, and better preservation of learned representations during scaling.

The introduction of the \kappa parameter provides a tunable mechanism for determining the ideal point at which to scale the model, balancing the trade-off between extracting maximum performance from smaller models and leveraging the benefits of larger architectures. 

While our experiments were limited to a 64M parameter model due to computational constraints, the principles demonstrated here have potential implications for scaling much larger models. 

The combination of nGPT's efficiency gains and our fractional scaling technique opens new possibilities for developing LLMs that are not only larger but also more efficient and robust. 

Future work should focus on, applying this technique to larger model scales, investigating the optimal \kappa values for different model sizes and tasks, exploring the potential for scaling both model width and depth using this approach, and analyzing the long-term effects of this scaling method on model performance and generalization

In conclusion, our BUS method, built upon the foundations of nGPT and HyperCloning, presents a promising direction for the future of LLM development. By enabling more efficient and flexible scaling, we pave the way for the creation of more powerful and resource-efficient language models.


## References
[1] Scaling Smart
[2] nGPT

## Appendix

