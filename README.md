# ADE_Project

## Project Description

The study proposes an NLP pipeline to extract Adverse Drug Events (ADEs) from unstructured text using the ADE corpus dataset. It consists of two main components: a Named Entity Recognition (NER) module to identify drugs and a Relationship Extraction (RE) module to link adverse events with drugs. The study compares the performance of general knowledge transformers like BERT with domain-specific ones like BioBERT and PubMedBERT to assess their effectiveness in this task. Evaluation metrics include precision, recall, and F1 score to measure the models' performance comprehensively.

In the NER task, the goal is to identify mentions of drugs within the text. This involves employing various autoencoding models, such as BERT, BioBERT, and PubMedBERT, and utilizing conventional IOB tags for annotation. The NER module aims to accurately identify drug mentions and label them accordingly.

In the RE task, the objective is to extract relationships between drugs and adverse events mentioned in the text. An architecture is implemented to generate drug-aware embeddings, focusing on tokens mentioning the drug. These embeddings are then used to train a fully connected layer with a sigmoid function to classify each token as having an adverse effect entity related to the mentioned drug. The RE module aims to effectively link adverse events with specific drugs mentioned in the text, enhancing performance through drug-aware sentence embeddings.

IOB tagging:

* **0: Outside**  - Represents tokens that are not part of any named entity.
* **1: B-drug** - Indicates the beginning of a drug named entity.
* **2: I-drug**   - Represents tokens that are inside a previously started drug named entity.
* **3: B-effect** - Indicates the beginning of an adverse effect named entity.
* **4: I-effect**   - Represents tokens that are inside a previously started adverse effect named entity

The project uses the The ADE-Corpus-V2 from https://huggingface.co/datasets/ade_corpus_v2


## Project Organization

The main.py calls the modules associated with each task, with each task residing in a separate file following an structure.


### Folder Structure 

* config.py: contains configurations parameters for the different models.

* data_process.py: process the dataset to create IOB tags, extract unique sentences and tokenize.

* eval.py: This file has different functions to evaluate the performance of the system

* training_NER.py: training for Name entity recognition of drugs and effects

* training_RE.py: training for Relation extraction, includes the model design. 

### Other Files

* environment.yml: file for the conda environment  configuration, useful for creating the required envrironment for this project.

* requirements.txt: list the packages that needs to be install for the environment.


Run conda env create -f environment.yml && conda activate NLP && python main.py this will run the main file after creating the environment and installing the required libraries.


