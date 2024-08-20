# Transfer learning transformers

The purpose of this script is to determine if there is a way to transfer knowledge learned from a smaller transformer model to a larger one. This could allow for more efficient training of LLM's since a large cost of triaining LLM's is the massive ammount of matmul, that need to be done in pretraining. If there exists a way to pretrain a smaller model and then expand it to a larger model, it could save time and money in pretraining costs.

## Background idea

### Principal Component Analysis:

In linear algebra there is a way to condense large matrices into smaller ones using Principal Component Analysis (PCA). More formally, this is a linear dimensionality reduction. Since there is a way to go from large to small, is there a way to go from small to large? It will never be able to perfectly recreate what the larger matrix should have been but, would it be possible for it to at least not start from the very begining? Is there some way to transfer that information that the smaller matrix has to the larger one?






# Steps 
- attempted doubling every col and row
- adding one hot vectors to each matrix to expand it that way
    - this works, knowlede from a 
    - test protocol:
        - Train a transformer for 5 epochs at half the size n/2
        - expand the transformer to full size by simply adding it to the n sized identity
        - train for an additional 5 epochs
        - train a seperate model at full size n for 10 epochs
        - compare average dev set accuracy
        - this test was performed 100 times and averaged to obtain a result accuracy of +/- 1%
    - results:
        - with smaller models that acheive worse performance, there seems to be little to no improvement, this is with matrix sizes of <48
        - with larger models, the improvement is larger

| Model size | Dev set accuracy |
|:----------:|:----------------:|
| 
 



## Personal Notes

- if generally a transformer learns via gradient descent, and you are training two models on the same problem, if one model is smaller than the other then isn't the smaller model learning a projection of the problem space in a lower dimension?
- if that is the case then the there could be some overlap in the solutions that can be found as you scale the model up, since it'll get a higher resolution image of the terrain of the problem space
- for example, if you take a paraboloid and project it onto a 2d plane, you can get different perspectives of the problem, some useful and some not so useful, but of the useful ones, the low dimensional solution does share some information with the full dimensional solution
- what does it mean for two matrices to have the same singular values?
- Since these matrices all seem to converge to the same singular values there must be something that they are all converging to 
- If they share singular values does that mean that we can construct a new larger matrix based on the singular values of the smaller matrix that will be closer to the solution for the larger matrix?




