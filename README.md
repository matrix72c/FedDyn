# Intro

An implementation of the [FedDyn: A dynamic and efficient federated distillation approach on Recommender System](https://ieeexplore.ieee.org/abstract/document/10077950).

# Citing this work

If you use this repository for academic research, you are encouraged to cite our paper:

```
@inproceedings{DBLP:conf/icpads/JinCGL22,
  author       = {Cheng Jin and
                  Xuandong Chen and
                  Yi Gu and
                  Qun Li},
  title        = {FedDyn: {A} dynamic and efficient federated distillation approach
                  on Recommender System},
  booktitle    = {28th {IEEE} International Conference on Parallel and Distributed Systems,
                  {ICPADS} 2022, Nanjing, China, January 10-12, 2023},
  pages        = {786--793},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/ICPADS56603.2022.00107},
  doi          = {10.1109/ICPADS56603.2022.00107},
  timestamp    = {Thu, 21 Sep 2023 13:15:57 +0200},
  biburl       = {https://dblp.org/rec/conf/icpads/JinCGL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Usage

Experiments were run using Python 3.10. To install dependencies:

```shell
pip install -r requirements.txt
```

To track the training process, run:

```shell
aim init && aim up
```

To run the experiments, modify the `config.yaml` and run:

```shell
python main.py
```
