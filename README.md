# Keras Demonstration

To run `keras_word_embeddings.py` please perform the following steps: <br>
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz <br>
!tar -xf aclImdb_v1.tar.gz

This will download and unzip the aclImdb database. **For this network this may not be the best usage.
The train dataset holds 3 claseses (positiv, negativ and undicided), but the test set only contains positive and negative reviews.**

But the code is not there to make any good performance, but to show how you can make: <br>
1) a normal neural network - code in `keras_network.py`
2) a neural network for multi class classification - code in `keras_word_embeddings.py`

Both networks have been written in different ways how you can initialize a neural network in Keras.
If you have any questions, feel free to ask. I have added so called conv layers, which could be removed without any real problem.
While this works, the results are only "ok" at best, but should give you a good point to go further.

Also you can review [Named Entity Recognition with Transformers](https://keras.io/examples/nlp/ner_transformers/), which is basically what you need / want to do.
But this utilizes so called "Transformer Networks", which I am not really familiar with.
But they are - as far as I know - the current state of the art, when it comes to NLP.

If you don't like Keras, please tell me & I will try my best to give you an example in PyTorch.
PyTorch is - in my opionion - a little easier once understood, but a little harder to learn.