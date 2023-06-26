# DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome

The official implementation of our paper



DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome



## Contents

- [1. Introduction](#1-introduction)
- [2. Model and Data](#2-model-and-data)
- [3. Setup Environment](#3-setup-environment)
- [4. Quick Start](#4-quick-start)
- [5. Pre-Training](#5-pre-training)
- [6. Evaluation](#6-evaluation)
- [7. Citation](#7-citation)





## 1. Introduction

DNABERT-2 is a foundation model trained on large-scale multi-species genome that achieves the state-of-the-art performanan on $28$ tasks of the GUE benchmark. It replaces k-mer tokenization with BPE, positional embedding with Attention with Linear Bias (ALiBi), and incorporate other techniques to improve the efficiency and effectiveness of DNABERT.



## 2. Model and Data

The pre-trained models is available at Huggingface as `zhihan1996/DNABERT-2-117M`. [Link]().



### 2.1 GUE: Genome Understanding Evaluation

GUE is a comprehensive benchmark for genome understanding consising of $28$ distinct datasets across $7$ tasks and $4$ species. GUE can be download [here](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view?usp=sharing). Statistics and model performances on GUE is shown as follows:



![GUE](figures/GUE.png)



![Performance](figures/Performance.png)



## 3. Setup environment

    # create and activate virtual python environment
    conda create -n dna python=3.8
    conda activate dna
    
    # install required packages
    python3 -m pip install -r requirements.txt





## 4. Quick Start

Our model is easy to use with the [transformers](https://github.com/huggingface/transformers) package.



```python
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
```





## 5. Pre-Training

Codes for pre-training is coming soon.





## 6. Evaluation





## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue or email Zhihan Zhou (zhihanzhou2020@u.northwestern.edu).



If you use DNABERT-2 in your work, please cite our paper:

```

```

