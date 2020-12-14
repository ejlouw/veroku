<div align="center">
  <img src="logo.svg">
</div>

[comment]: # (doc-start)

![Build Status](https://github.com/ejlouw/veroku/workflows/CI_PIPELINE/badge.svg?branch=master)

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
It is recommended to use a separate conda virtual environment when installing the dependencies, to avoid interfering
with existing packages. To get started with conda environments, see the
[installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
For more information on using conda environments see
[managing environments guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Overview
<div style="text-align: justify">
Veroku is an open source library for building and performing inference with probabilistic graphical models (PGMs) in
python. PGMs provide a framework for performing efficient probabilistic inference with very high dimensional
distributions. A typical example of a well-known type of PGM is the Kalman filter that can be used to obtain
probabilistic estimates of a hidden state of a process or system, given noisy measurements. PGMs can in principle be
used for any problem that involves uncertainty and is therefore applicable to many problems.</div> 

Veroku currently supports the following probability distributions:
* Categorical (sparse and dense implementations)
* Gaussian
* Gaussian mixture<sup>1</sup>
* Linear Gaussian<sup>2</sup>
* Non-linear Gaussian<sup>3</sup>

<sup>1</sup> This class still has some experimental functionality (specifically the Gaussian mixture division methods)
and is, therefore, still in the factors.experimental sub-package.  
<sup>2</sup> Using the Gaussian class - see the Kalman filter example notebook.<br/>
<sup>3</sup>This implementation is still experimental (see the factors.experimental sub-package).


<div style="text-align: justify">
These distributions can be used as factors to represent a factorised distribution. These factors can be used, together
with the <code>cluster_graph</code> module to automatically create valid cluster graphs. Inference can be performed in these graphs
using message passing algorithms. Currently only the LBU (Loopy Belief Update) message-passing algorithm is supported.
</div>

<br/>
Example notebooks:

* [Toy example](https://github.com/ejlouw/veroku/blob/master/examples/slip_on_grass.ipynb)
* [Kalman filter](https://github.com/ejlouw/veroku/blob/master/examples/Kalman_filter.ipynb)
* [Sudoku](https://github.com/ejlouw/veroku/blob/master/examples/sudoku.ipynb)



### On the Roadmap
The following distributions, models and features are on the roadmap to be added to veroku:
* Conditional Gaussian
* Dirichlet distribution
* Wishart distribution
* Normal-Wishart distribution
* Plate models
* Structure Learning
* Causal Inference

### Dependencies
For the python dependencies see the [requirements](https://github.com/ejlouw/veroku/blob/master/requirements.txt) file.
The following additional dependencies are also required for some functionality (these are not installed automatically
 with the `pip install`):

##### Graphviz
See https://graphviz.org/download/ for installation instructions. 

### Contributing
If you would like to contribute to veroku, please see the [contributing guide](https://github.com/ejlouw/veroku/blob/master/contributing.md).

### License
Veroku is released under a 3-Clause BSD license. You can view the license
[here](https://github.com/ejlouw/veroku/blob/master/LICENSE).
