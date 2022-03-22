# Automatic Code Summarization
This project is a part of the Natural Language Processing course (COMP 8730).


## Project Introduction/ Objective
The purpose of this project is to aid the process of code documentation by generating summary of a code snippet written in Python, Java, JavaScript, and PHP. In this
project, we explore the following two approaches:
* An hierarchical multi-lingual code-summarization using Bi-directional Encoder Representations from Transformer (BERT) for classification and Bidirectional Auto Regressive Trans-former (BART) for summarization.
* Multi-lingual code summarization using BART.


### Hierarchical code summarization
This approach uses a two level modelling approach with the first layer being a classifier built using BERT, to detect the programming language of the code snippet. Next, based on the language detected, a fine-tuned BART model is used to generate the summary of the source code. Each BART model is trained on only one language. We have used HuggingFace library for pre-trained BERT ([bert-base-uncased](https://huggingface.co/bert-base-uncased)) and BART ([ncoop57/bart-base-code-summarizer-java-v0](https://huggingface.co/ncoop57/bart-base-code-summarizer-java-v0)) models. 

###  Multi-lingual code summarization
In this approach, we have trained a single BART model on code snippets from Python, Java, JavaScript, and PHP languages and their corresponding summaries which are tokenizaed using BART tokenizer. 

## Data
The models are trained on [CodeSearchNet](https://github.com/github/CodeSearchNet) dataset which contains code snippets and their corresponding summaries for Python, Java, JavaScript, PHP, Ruby and Go. As an experiment, we have considered Python, Java, JavaScript, and PHP for training our models.


## Install
Run the following command to install the required dependencies

```bash
pip install -r requirements.txt
```

## Evaluation
We used BLUE(BiLingual evaluation understudy), and ROGUE(Metric for Evaluation of Translation with Explicit ORdering) score to evaluate the performance of the model.  

## Flow of the project
Different folders hold code for different sections of the pipeline.

##### Experiment folder
The wrappers to use the pipeline to train the Hierarchical Code Summarization Model and Multilingual summarization model

##### Flow folder
It holds the flow of hierarchical and multilingual approach.
  - Hierarchical - A class with all the functionalities of a Heirarchical Summarization model. Objects of this class provide functionality like train, prediction and  evalution,making it worthy of being called a model.
  - Mulitlingual- A class with all the functionalities of a multilingual Summarization model.Objects of this class provide functionality like train, prediction and      evalution,making it worthy of being called a model.
  - metrices - to compute metrices score like bleu and rogue.

##### Models folder
It contains two huggingface model one is BERT for classification and another is BART for sumarization.






