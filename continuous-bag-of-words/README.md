# continuous-bag-of-words
The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word.
This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typically, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost always helps performance a couple of percent.

This is the solution of the final exercise of [this](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html) great tutorial on NLP in PyTorch.

#### Example
Corpus
``` 
We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.
```
Context
```
People, create, to, direct
```
Output
```
programs
```
