# DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome

The repo contains: 

1. The official implementation of [DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome](https://arxiv.org/abs/2306.15006)
2. Genome Understanding Evaluation (GUE): a comprehensize benchmark containing 28 datasets for multi-species genome understanding benchmark.



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

The pre-trained models is available at Huggingface as `zhihan1996/DNABERT-2-117M`. [Link to HuggingFace ModelHub](https://huggingface.co/zhihan1996/DNABERT-2-117M). [Link For Direct Downloads]().



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


To load the model from huggingface:
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
```


To calculate the embedding of a dna sequence
```
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape) # expect to be 768
```



## 5. Pre-Training

Codes for pre-training is coming soon.





## 6. Evaluation

### 6.1 Evaluate models on GUE
Please first download the GUE dataset from [here](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view?usp=sharing). Then run the scripts to evaluate on all the tasks. 

Current script is set to use `DataParallel` for training on 4 GPUs. If you have different number of GPUs, please change the `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly to adjust the global batch size to 32 to replicate the results in the paper. If you would like to perform distributed multi-gpu training (e.g., with `DistributedDataParallel`), simply change `python` to `torchrun --nproc_per_node ${n_gpu}`.


```
export DATA_PATH=/path/to/GUE #(e.g., /home/user)
cd finetune

# Evaluate DNABERT-2 on GUE
sh scripts/run_dnabert2.sh DATA_PATH

# Evaluate DNABERT (e.g., DNABERT with 3-mer) on GUE
# 3 for 3-mer, 4 for 4-mer, 5 for 5-mer, 6 for 6-mer
sh scripts/run_dnabert1.sh DATA_PATH 3

# Evaluate Nucleotide Transformers on GUE
# 0 for 500m-1000g, 1 for 500m-human-ref, 2 for 2.5b-1000g, 3 for 2.5b-multi-species
sh scripts/run_nt.sh DATA_PATH 0

```

### 6.2 Evaluate on your own datasets
Comming soon.


## 7. Citation

If you have any question regarding our paper or codes, please feel free to start an issue or email Zhihan Zhou (zhihanzhou2020@u.northwestern.edu).



If you use DNABERT-2 in your work, please kindly cite our paper:

**DNABERT-2**

```
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```

**DNABERT**

```
@article{ji2021dnabert,
    author = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume = {37},
    number = {15},
    pages = {2112-2120},
    year = {2021},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab083},
    url = {https://doi.org/10.1093/bioinformatics/btab083},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/37/15/2112/50578892/btab083.pdf},
}
```

