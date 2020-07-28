# CyberWallE at SemEval-2020 Task 11: An Analysis of Feature Engineering for Ensemble Models for Propaganda Detection

With the advent of rapid dissemination of news articles through online social media, automatic detection of biased or fake reporting has become more crucial than ever before.
This repository contains the code and article describing our participation in both subtasks of the SemEval 2020 shared task for the [Detection of Propaganda Techniques in News Articles](https://propaganda.qcri.org/semeval2020-task11/).

The Span Identification (SI) subtask is a binary classification problem to discover propaganda at the token level, and the Technique Classification (TC) subtask involves a 14-way classification of propagandistic text fragments.
We use a bi-LSTM architecture in the SI subtask and train a complex ensemble model for the TC subtask.
Our architectures are built using embeddings from BERT in combination with additional lexical features and extensive label post-processing.
Our systems achieve a rank of 8 out of 35 teams in the SI subtask (F1-score: 43.86%) and 8 out of 31 teams in the TC subtask (F1-score: 57.37%).

Our [article](https://github.com/cicl-iscl/CyberWallE-propaganda-detection/blob/master/Final%20Paper%20Submission/CyberWallE_2020.pdf) provides an extensive exploration of various embedding, feature and classifier combinations.
The repository is organized as follows:

- `baselines` (from the organizers, empty in the remote): Baseline code + predictions
- `data` (empty in the remote*): Training/development input files with features, lexica for semantic + rhetorical structures (*Some of the contents can be downloaded from sources given in the folder, the rest can be generated using the files in `utils`)
- `datasets` (from the organizers, empty in the remote): Articles, training labels
- `eda`: Code for analyzing label distributions, sentence lengths and other features of the given data
- `models`: Our models
- `tools` (from the organizers, empty in the remote): Scripts for evaluating the data
- `utils`: Code for data pre- and post-processing and evaluation

```
@InProceedings{SemEval2020-11-CyberWallE,
author = "Blaschke, Verena and Korniyenko, Maxim and Tureski, Sam",
title = "{CyberWallE} at {SemEval}-2020 {T}ask 11: An Analysis of Feature Engineering for Ensemble Models for Propaganda Detection",
pages = "",
booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation",
series = "SemEval 2020",
year = "2020",
address = "Barcelona, Spain",
month = "December",
}
```
