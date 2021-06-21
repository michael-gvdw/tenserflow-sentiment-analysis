# Sentiment Analysis w/ Tensorflow

## Project Overview

### Context

For the Open Program (Week 17 - Week 18) I will spend my time investigating Sentiment Analysis. Sentiment Analysis is the use of natural language processing, text analysis. computational liguistics, and biometrics to systematicaly identify, extract, quatify, and study effective states and subjective information. In essence this means to extract meaning from language with the use of a machine.

I will follow the official [Tensorflow tutorial](https://www.tensorflow.org/text/tutorials/text_classification_rnn).

### Goal

I will spend time investigating Sentiment Analysis in order to classify a sentence either negative, neutral or positive with the use of Deep Learing.

## Data

Tenserflow provides its own datasets for experimenting and learning. For this project I will use the "imdb_reviews" dataset. This dataset is for binary sentiment classification containing a lot of data. Provided are 25.000 higly polar movie reviews for training and 25.000 for testing.

## Data Preparation

Before continuing onto the modelling phase there are two things that need to take place beforehand:

1) word embeddings
2) padding sentences

### Word embedding

In Natural Language Processing, word embedding is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning. In summary words are reprented as vectors in n-dimensions and the distance between the vectors show how much each word correlates to eachother. 

The "imdb_reviews" dataset provides an encoder along with the data we can use to encode the words. 

### Padding sentences

Before passing the senctence to the model for training all the sentences (inputs) need to be of same size. To achieve this a sequence of zeros will be added to the end of a sentence until it reaches the same length as the sentence with the max length.

## Modelling 
<!-- ![](https://www.tensorflow.org/text/tutorials/images/bidirectional.png) -->

1) The model can be build as a `tf.keras.Sequential` since all layers in the model only have single input and single output. `tf.keras.Sequential` groups a linear stack of layers into a `tf.keras.Model`.

2) First is an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequence of words indices to sequences of vectors. These vectors are trainable. After training, words with similar meanings often have similar vectors.

3) A Recurrent Neural Network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input of the next timestep.

    The `tf.keras.layers.Bidirectional` wrapper can be used with RNN layer. This propagates the input foward and backwards through the RNN layer and then concatenates the final output.

    ![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

4) After the RNN has converted the sequence to a single vector the two `tf.keras.layers.Dense` do some final processing, and convert this vector representation to a single value as the classification output.

Below is the code implementation of the above description:

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

The model is capable of achieving 87% accuracy. Which is good given the time and scope of the project.

## Conclusion

In summary I have succesfully created a Deep Learning Model that is capable of giving a percetage of how positive a sentence is. I am pleased with the current state of the project as I have been able to aquire basic knowledge of how the process is executed and conducted. 


