# Federated-Linear-Contextual-Bandits
This repository is the official implementation of *Huang, R., Wu, W., Yang, J., & Shen, C. (2021). Federated linear contextual bandits. Advances in neural information processing systems, 34, 27057-27068.*
## Requirements
-  Programming language: *Python3*,
-  Packages *Numpy 1.17.4, Scipy 1.3.2, Pandas 0.25.3*.

## Usage
To get a preliminary result, run this command:
```
python3 main.py
```
It will plot the regret per client *R(T)* as function of *T* of four algorithms specified in the paper, i.e. Fed-PE, Local UCB, Enhanced Fed_PE, and Collaborative. The last algorithm is a modified version for full information exchange. 

The feature vectors and arms parameters $\theta$ are generated in SyntheticProblem.py, which is a synthetic dataset.

To run the experiment on the MovieLens-100K, replace 
```python 
import SyntheticProblem as Construction
```
by
```python
import MovieLensProblem as Construction
```
in the file main.py. Then run the command:
```
python3 main.py
```
It will plot the regret per client on the MovieLens-100K dataset.

## Completing MovieLens-100K
The complete rating matrix is stored in complete_ratings.csv. The file can be get by running this command:
```
python3 CollaborativeFilteringCopy.py
```
CollaborativeFilteringCopy.py is directly modified from the Github project: *Collaborative-Filtering* (https://github.com/kevalmorabia97/Collaborative-Filtering).

MovieLens-100K (Harper, F.M. and Konstan, J.A. (2015) The MovieLens Datasets: History and context. ACM Trans. Intract. Intell. Syst., 5(4).)

## Optimization Problem
To solve multi-client G-optimal design or its equivalent Determinant Maximizaion subject to multi-constraints, use the function `OptimalExperiment` in minVar.py.
