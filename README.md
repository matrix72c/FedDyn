# Intro

An implementation of the [FedDyn: A dynamic and efficient federated distillation approach on Recommender System](https://ieeexplore.ieee.org/abstract/document/10077950).

# Citing this work

If you use this repository for academic research, you are encouraged to cite our paper:

```
@INPROCEEDINGS{10077950,
  author={Jin, Cheng and Chen, Xuandong and Gu, Yi and Li, Qun},
  booktitle={2022 IEEE 28th International Conference on Parallel and Distributed Systems (ICPADS)}, 
  title={FedDyn: A dynamic and efficient federated distillation approach on Recommender System}, 
  year={2023},
  volume={},
  number={},
  pages={786-793},
  doi={10.1109/ICPADS56603.2022.00107}}
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