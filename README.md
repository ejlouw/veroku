<div align="center">
  <img src="logo.svg">
</div>

[comment]: # (doc-start)

### Disclaimer
This project is still in the very early stages of development. Use this library at your own risk. The github repo is also still private,
but will be made public soon.

### Installation
For installing through pip:
```bash
pip install veroku
```

To clone this git repo:
```
git clone https://github.com/ejlouw/veroku.git 
cd veroku/
pip install -r requirements.txt
```

### Overview
<div style="text-align: justify">
Veroku is an open source library for building and performing inference with probabilistic graphical models (PGMs). PGMs
provide a framework for performing efficient probabilistic inference with very high dimensional distributions. A typical
example of a well-known type of PGM is the Kalman filter that can be used to obtain probabilistic estimates of a hidden
state of a process or system, given noisy measurements. PGMs can in principle be used for any problem that involves
uncertainty and is therefore applicable to many problems.</div> 

Veroku currently supports the following distributions:
* Categorical
* Gaussian
* Gaussian mixture
* Linear Gaussian<sup>1</sup>
* Non-linear Gaussian<sup>2</sup>

<sup>1</sup> Using the Gaussian class - see the Kalman filter example notebook.<br/>
<sup>2</sup>This implementation is still experimental.


<div style="text-align: justify">
These distributions can be used as factors to represent a factorised distribution. These factors can be used, together
with the `cluster_graph` module to automatically create valid cluster graphs. Inference can be performed in these graphs
using message passing algorithms. Currently only the LBU (Loopy Belief Update) message-passing algorithm is supported.
</div>

<br/>
Example notebooks:

* [Toy example](https://github.com/ejlouw/veroku/blob/master/examples/slip_on_grass.ipynb)
* [Kalman filter](https://github.com/ejlouw/veroku/blob/master/examples/Kalman_filter.ipynb)
* [Sudoku](https://github.com/ejlouw/veroku/blob/master/examples/sudoku.ipynb)


### Future Features
To be added soon:
* More example notebooks
* Plate models (for efficiently specifying PGMs as modular/hierarchical templates)

On the roadmap:
* Dirichlet distribution
* Wishart distribution
* Normal-Wishart distribution

### License
Veroku is released under 3-Clause BSD license. You can view the license at [here](https://github.com/ejlouw/veroku/blob/master/LICENSE).
