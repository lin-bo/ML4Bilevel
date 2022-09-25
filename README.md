# ML4Bilevel

This is a repository for [A Machine Learning Approach to Solving Large Bilevel and Stochastic Programs: Application to Cycling Network Design](https://arxiv.org/abs/2209.09404).

## Dependencies
- [Gurobi](https://www.gurobi.com)
- [Numpy](https://numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Scikit Learn](https://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org)
- [tqdm](https://tqdm.github.io)
- [Scipy](https://scipy.org)
- [Gensim](https://radimrehurek.com/gensim/)

## Preparation

## Synthetic Instances

## Real Instances

We provide code to re-produce the case-study results. However, important note that the calculation requires weeks to finish. 
You can access the pre-calculated results for the first two steps in `./prob/trt/res/job/precalculated/`.

Step 0: download data:
- Network instances, put the two files under `./prob/trt/`
- OD pair embedding, put the file under `./prob/trt/emb/`

Step 1: To obtain network designs, run
```commandline
python solve_trt_opt_cmd.py\
 -s 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41\
 -b 40 80 120 160 200 240 280 320 360 400\
 --n 2000\
 --potential job  
```

Here we solve the *k*NN-augmented model with 21 different *p*-median samples (each consists of 2000 OD pairs) 
and 10 different budgets (10, 20, ..., 100 km) using destination job count as the potential for accessibility calculation. 


Step 2: To calculate the accessibility of each network design, run
```commandline
python 
```

## Citation
```
@article{chan2022machine,
  title={A Machine Learning Approach to Solving Large Bilevel and Stochastic Programs: Application to Cycling Network Design},
  author={Chan, Timothy C. Y. and Lin, Bo and Saxe, Shoshanna},
  journal={arXiv preprint arXiv:2209.09404},
  year={2022}
}
```
