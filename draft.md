# Gradual Smart Scaling for Accelerated LLM Pre-training with small model initalization 


## Abstract
Due to the quadratic complexity of self attention, some methods have been developed to use smaller models as initalizations for training larger Large Language Models (LLMs) as described in [1].

This method however requires a doubling of the model size, in this paper we show that this is not strictly neccesary and that a fractional scale up and provide more robust model performance.

Additionally, there have been recent breakthroughs that can also drastically reduce pre-training time by using normalization techniques that acheive 4-20x convergence speed up [2]. 

This method has been used to reach a models capacity. 

Our methods show that it is not required to train a model to maximum performance in order to scale up a model and that the scaling up procedure can be incremental to preserve performance in the scaling process.


## Introduction
- Linformer
- Smart Scaling
- Scaling laws
- nGPT

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

