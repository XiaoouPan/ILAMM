# ILAMM 

Nonconvex Regularized Robust Regression via I-LAMM (**I**terative **L**ocal **A**daptive **M**ajorize-**M**inimization) Algorithm

## Description

This package employs the I-LAMM algorithm to solve regularized Huber regression. The choice of penalty functions includes the l<sub>1</sub>-norm, the smoothly clipped absolute deviation (SCAD) and the minimax concave penalty (MCP). Two tuning parameters lambda and tau (for Huber loss) are calibrated by cross-validation. As a by-product, this package also produces regularized least squares estimators, including the Lasso, SCAD and MCP. See the reference papers for more details. 

Assume that the observed data (Y, X) follow a linear model Y = X &beta; + &epsilon;, where Y is an n-dimensional response vector, X is an n by d design matrix, &beta; is a sparse vector and &epsilon; is an n-vector of noise variables whose distributions can be asymmetric and/or heavy-tailed. The package will compute the regularized Huber regression estimator.

## Installation

Install `ILAMM` from GitHub:

```{r gh-installation, eval = FALSE}
install.packages("devtools")
library(devtools)
devtools::install_github("XiaoouPan/ILAMM")
library(ILAMM)
```

## Getting help

Help on the functions can be accessed by typing `?`, followed by function name at the R command prompt. 

For example, `?ncvxHuberReg` will present a detailed documentation with inputs, outputs and examples of the function `ncvxHuberReg`.

## Common error messages

The package `ILAMM` is implemented in `Rcpp` and `RcppArmadillo`, so the following error messages might appear when you first install it (we'll keep updating common error messages with feedback from users):

* Error: "...could not find build tools necessary to build ILAMM": For Windows you need Rtools, for Mac OS X you need to install Command Line Tools for XCode. See [this link](https://support.rstudio.com/hc/en-us/articles/200486498-Package-Development-Prerequisites) for details. 

* Error: "library not found for -lgfortran/-lquadmath": It means your gfortran binaries are out of date. This is a common environment specific issue. 

    1. In R 3.0.0 - R 3.3.0: Upgrading to R 3.4 is strongly recommended. Then go to the next step. Alternatively, you can try the instructions [here](http://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/).

    2. For >= R 3.4.* : download the installer from [here](https://gcc.gnu.org/wiki/GFortranBinaries#MacOS). Then run the installer.


## Functions

There are five functions, all of which are implemented by I-LAMM algorithm. 

* `ncvxReg`: Nonconvex regularized regression (Lasso, SCAD, MCP). 
* `ncvxHuberReg`: Nonconvex regularized Huber regression (Huber-Lasso, Huber-SCAD, Huber-MCP).
* `cvNcvxReg`: K-fold cross-validation for nonconvex regularized regression.
* `cvNcvxHuberReg`: K-fold cross-validation for nonconvex regularized Huber regression.
* `tfNcvxHuberReg`: Tuning-free nonconvex regularized Huber regression.

## Examples 

Here we generate data from a sparse linear model Y = X * beta + epsilon, where beta is sparse and epsilon consists of indepedent coordinates from a log-normal distribution, which is asymmetric and heavy-tailed. 

```{r}
library(ILAMM)
n = 50
d = 100
set.seed(2018)
X = matrix(rnorm(n * d), n, d)
beta = c(rep(2, 3), rep(0, d - 3))
Y = X %*% beta + rlnorm(n, 0, 1.2) - exp(1.2^2 / 2)
```

Fist, we apply Lasso to fit (Y, X) as a benchmark method. It can be found that Lasso overfits the data, which means the estimated signal it produces has high false positive.

```{r}
fitLasso = cvNcvxReg(X, Y, penalty = "Lasso")
betaLasso = fitLasso$beta
```

Then we apply linear regression with non-convex penalties to fit (Y, X): SCAD and MCP. We can find advantages of SCAD and MCP over Lasso, since non-convex penalties reduce the biasedness introduced by l1 penalty.

```{r}
fitSCAD = cvNcvxReg(X, Y, penalty = "SCAD")
betaSCAD = fitSCAD$beta
fitMCP = cvNcvxReg(X, Y, penalty = "MCP")
betaMCP = fitMCP$beta
```

We further apply Huber regression with non-convex penalties to fit (Y, X): Huber-SCAD and Huber-MCP. With heavy-tailed sampling, we can see evident advantages of Huber-SCAD and Huber-MCP over their least squares counterparts, SCAD and MCP.

```{r}
fitHuberSCAD = cvNcvxHuberReg(X, Y, penalty = "SCAD")
betaHuberSCAD = fitHuberSCAD$beta
fitHuberMCP = cvNcvxHuberReg(X, Y, penalty = "MCP")
betaHuberMCP = fitHuberMCP$beta
```

Finally, we demonstrate non-convex regularized Huber regression with tau calibrated via a tuning-free principle. This function is computationally more efficient, as we decrease cross-validation from two-dimensional grid search to one-dimension. See the reference paper (Wang et al., 2018) for more details of the tuning-free procedure.

```{r}
fitHuberSCAD.tf = tfNcvxHuberReg(X, Y, penalty = "SCAD")
betaHuberSCAD.tf = fitHuberSCAD$beta
fitHuberMCP.tf = tfNcvxHuberReg(X, Y, penalty = "MCP")
betaHuberMCP.tf = fitHuberMCP$beta
```

## Notes 

Function `cvNcvxHuberReg` might be slow, because it carries out a two-dimensional grid search to choose lambda and tau using cross-validation.

## License

GPL (>= 2)

## Authors

Xiaoou Pan <xip024@ucsd.edu>, Qiang Sun <qsun@utstat.toronto.edu>, Wen-Xin Zhou <wez243@ucsd.edu> 

## Reference

Eddelbuettel, D. and Francois, R. (2011). Rcpp: Seamless R and C++ Integration. J. Stat. Softw. 40(8) 1-18. [paper](http://dirk.eddelbuettel.com/code/rcpp/Rcpp-introduction.pdf)

Eddelbuettel, D. and Sanderson, C. (2014). RcppArmadillo: Accelerating R with high-performance C++ linear algebra. Comp. Stat. Dat. Ana. 71 1054-1063. [paper](http://dirk.eddelbuettel.com/papers/RcppArmadillo.pdf)

Fan, J. and Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. J. Amer. Statist. Assoc. 96 1348–1360. [paper](https://www.tandfonline.com/doi/abs/10.1198/016214501753382273)

Fan, J., Li, Q. and Wang, Y. (2017). Estimation of high dimensional mean regression in the absence of symmetry and light tail assumptions. J. R. Stat. Soc. Ser. B. Stat. Methodol. 79 247-265. [paper](https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssb.12166)

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841. [paper](https://projecteuclid.org/euclid.aos/1522742437)

Huber, P. J. (1964). Robust estimation of a location parameter. Ann. Math. Statist. 35 73–101. [paper](https://projecteuclid.org/euclid.aoms/1177703732)

Pan, X., Sun, Q. and Zhou, W.-X. (2019). Nonconvex regularized robust regression with oracle properties in polynomial time. Preprint. [paper](https://arxiv.org/abs/1907.04027).

Sanderson, C. and Curtin, R. (2016). Armadillo: a template-based C++ library for linear algebra. J. Open. Src. Softw. 1 26. [paper](http://conradsanderson.id.au/pdfs/sanderson_armadillo_joss_2016.pdf)

Sun, Q., Zhou, W.-X. and Fan, J. (2018) Adaptive Huber regression, J. Amer. Statist. Assoc, to appear. [paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1543124)

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Stat. Soc. Ser. B. Stat. Methodol. 58 267–288. [paper](https://www.jstor.org/stable/2346178?seq=1#metadata_info_tab_contents)

Wang, L., Zheng, C., Zhou, W. and Zhou, W.-X. (2018). A new principle for tuning-free Huber regression. Preprint. [paper](https://www.math.ucsd.edu/~wez243/Tuning_Free.pdf)

Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. Ann. Statist. 38 894–942. [paper](https://projecteuclid.org/euclid.aos/1266586618)
