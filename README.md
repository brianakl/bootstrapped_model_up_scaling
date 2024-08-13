# Transfer learning transformers

The purpose of this script is to determine if there is a way to transfer knowledge learned from a smaller transformer model to a larger one. This could allow for more efficient training of LLM's since a large cost of triaining LLM's is the massive ammount of matmul, that need to be done in pretraining. If there exists a way to pretrain a smaller model and then expand it to a larger model, it could save time and money in pretraining costs.

## Background idea

### Principal Component Analysis:

In linear algebra there is a way to condense large matrices into smaller ones using Principal Component Analysis (PCA).
