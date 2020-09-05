<div align="center">
  <img src="veroku_logo.png">
</div>

### Overview
<div style="text-align: justify">
Veroku is an open source library for building and performing inference with probabilistic graphical models (PGMs). PGMs
provide a framework for performing efficient probabilistic inference with very high dimensional distributions. A typical
example of a well-known type of PGM is the Kalman filter that can be used to obtain probabilistic estimates of a hidden
state of a process or system, given noisy measurements. PGMs can in principle be used for any problem that involves
uncertainty and is therefore applicable to many problems.</div> 

Currently, veroku supports the following distributions:
* Multivariate Gaussian
* Multivariate Gaussian mixture
* Multivariate categorical

<div style="text-align: justify">
These distributions can be used as factors to represent a factorised distribution. These factors can be used, together
with the `cluster_graph` module to automatically create valid cluster graphs. Inference can be performed in these graphs
using message passing algorithms. Currently only the LBU (Loopy Belief Update) message-passing algorithm is supported.
</div>

### Future Features

To be added soon:
* Example notebooks
* Non-linear Gaussian distribution
* Plate models (for efficiently specifying PGMs as modular/hierarchical templates)

On the roadmap:
* Dirichlet distribution
* Wishart distribution
* Normal-Wishart distribution

