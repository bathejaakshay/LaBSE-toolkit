# LaBSE-toolkit

#### What is LaBSE?
[Feng et al. (2020)](https://aclanthology.org/2022.acl-long.62.pdf) proposed the LaBSE model, which is a multilingual sentence embedding model trained on 109 languages, including some Indic languages.


#### This toolkit can perform following three tasks
  1. Assign quality scores to the Parallel Corpus
  2. Extract high-quality Parallel Corpus from the noisy Pseudo-Parallel corpus
  3. Perform Sentence Alignment on a given misaligned Parallel corpus.
  

#### Prerequisites

1. Install Sentence-Transformers  
`pip install sentence-transformers`    

2. Install pytorch with CUDA 11  
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`  

## Task 1: Assign quality-scores to the Parallel Corpus  
```
python LaBSE-toolkit/sent_align.py --source test-en.txt --target test-mr.txt --batch_size 1000 --operation score
```

## Task 2: Extract high-quality Parallel corpus from Pseudo-Parallel corpus.
### Hyperparameter: `threshold`  
The Parallel sentences are extracted based on the threshold quality score provided.  

```
python LaBSE-toolkit/sent_align.py --source test-en.txt --target test-mr.txt --batch_size 1000 --operation score --threshold 0.8
```

## Task 3: Sentence Alignment in Parallel Corpus  
### set `--operation` flag as `sent-align`  
### and set `--threshold` as per the required quality of parallel corpus  

```
python LaBSE-toolkit/sent_align.py --source test-en.txt --target test-mr.txt --batch_size 1000 --operation sent-align --threshold 0.8
```

### HELP-OPTIONS

```
-src, --source: PATH to source file
-tgt, --target: PATH to target file
-th, --threshold: LaBSE threshold value for extracting high quality data
-b, --batch_size: batch_size for LaBSE scoring 
-op, --operation: Select operation between score and sent-align
-mp, --model_path: Path to the saved LaBSE model

```

## The hands-on tutorial can be found in this [Colab Notebook](https://colab.research.google.com/drive/1jLrI8FDCiGP4KKfPqzl018-u1HclS4V4?usp=sharing) 


