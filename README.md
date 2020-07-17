# ASiNE: Adversarial Signed Network Embedding
This repository provides a reference implementation of *ASiNE* as described in the following paper:
> ASiNE: Adversarial Signed Network Embedding<br>
> Yeon-Chang Lee, Nayoun Seo, Kyungsik Han and Sang-Wook Kim<br>
> 43rd Int'l ACM Conf. on Research and Development in Information Retrieval (ACM SIGIR 2020)<br>

### Authors
- Yeon-Chang Lee (lyc0324@hanyang.ac.kr)
- Nayun Seo (nayounseo@hanyang.ac.kr)
- Kyungsik Han (kyungsikhan@ajou.ac.kr)
- Sang-Wook Kim (wook@hanyang.ac.kr)

### Input
The input files should be saved in `data/` folder. The structure of the input file is the following:

```node_id1 node_id2 sign```

Node ids start from *0* to *N-1* (*N* is the number of nodes in the graph).

### Output
The output files are saved in `results/` folder. The first line has the following format:

```num_of_nodes dim_of_embeddings```

The next *N* lines are as follows:

```node_id dim1 dim2 ... dimk```

where dim1, ... , dimk is the *k*-dimensional embedding vectors learned by *ASiNE*.

### Arguments

```
--dataset                 Dataset name (default: "wikirfa")
--n_emb                   Dimensionality of embedding. (default: 128)
--lr                      Learning rate (default: 0.01)
--window_size             Size of context window (default: 2)
--learn_fake_pos          Whether performing the additional learning of the fake positive edges generated from negative generator (default: False)
--n_epochs                Number of epochs to train (default: 20)
--n_epochs_gen            Number of iterations for the generator per epoch (default: 10)
--n_epochs_dis            Number of iterations for the discriminator per epoch (default: 10)
--n_sample_gen            Number of generating edges (default: 20)
--batch_size_gen          Batch size for the generator (default: 64)
--batch_size_dis          Batch size for the discriminator (default: 64)
--n_node_subsets          Number of subsets to divide nodes existing in the positive or negative graph for large datasets (default: 1)
--lambda_gen              L2 loss regularization weight for the generator (default: 0.00001)
--lambda_dis              L2 loss regularization weight for the discriminator (default: 0.00001)    
```

### Basic Usage
```
python ./src/main.py --dataset wikirfa --n_emb 128 --lr 0.01 --window_size 2 --learn_fake_pos True --n_epochs 20 --n_node_subsets 2  
```

### Requirements
The code has been tested running under Python 3.6. The required packages are as follows:

- ```tensorflow == 1.8.0```
- ```numpy == 1.14.5```
- ```pandas == 0.24.2```
- ```scikit-learn == 0.19.1```
- ```tqdm == 4.23.4```

### Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{lee20sigir,
  authors   = {Yeon{-}Chang Lee and Nayoun Seo and Kyungsik Han and Sang{-}Wook Kim},
  title     = {ASiNE: Adversarial Signed Network Embedding},
  booktitle = {International ACM SIGIR Conference on Research and Development in Information Retrieval (ACM SIGIR 2020)},      
  year      = {2020}
}
```