<div align="center">
  <img src="logo.svg">
</div>

[comment]: # (doc-start)

### Disclaimer
This project is still in the very early stages of development. Use this library at your own risk. The github repo is 
also still private, but will be made public soon.

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
* Conditional Gaussian
* Dirichlet distribution
* Wishart distribution
* Normal-Wishart distribution
* Structure Learning
* Causal Inference

### Dependencies
For the python dependencies see the [requirements](https://github.com/ejlouw/veroku/blob/master/requirements.txt) file.
The following additional dependencies are also required for some functionality (these are not installed automatically
 with the `pip install`):

##### Graphviz
See https://graphviz.org/download/ for installation instructions. 


### Contributing
<div style="text-align: justify">
If you would like to contribute to veroku, the TODO's in the code are a good place to start. There are a few very simple
 ones, but some are more complex. Another area where contributions will be valuable is in getting the experimental
 modules rounded off and writing tests for these modules. Another potential area for contribution would be the items on 
 the roadmap, although it would be best if the experimental modules are first rounded off.  In general, please remember
 to ensure that the following guidelines are followed when contributing:
</div>

* The use of pylint disable statements should be reserved only for special cases and are not generally acceptable.
* Add tests for any contributions ( this will also prevent the build from failing on the code coverage check)	


### License
Veroku is released under a 3-Clause BSD license. You can view the license
[here](https://github.com/ejlouw/veroku/blob/master/LICENSE).
