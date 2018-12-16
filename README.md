# ILAMM 

Nonconvex Regularized Robust Regression via I-LAMM (iterative local adaptive majorize-minimization) Algorithm

## Goal of this package

This package employs the I-LAMM algorithm to solve regularized Huber regression. The choice of penalty functions includes the l1-norm, the smoothly clipped absolute deviation (SCAD) and the minimax concave penalty (MCP). Two tuning parameters lambda and tau (for Huber loss) are calibrated by cross-validation. As a by-product, this package also produces regularized least squares estimators, including the Lasso, SCAD and MCP. See the reference papers for more details. 

The observed data are (Y,X), where Y is an n-dimensional response vector and X is an n by d design matrix. We assume that Y depends on X through a linear model Y = X * beta + epsilon, where epsilon is an n-dimensional noise vector whose distribution can be asymmetrix and/or heavy-tailed. The package computes the standard Huber's M-estimator if d < n and the regularized Huber regression estimator if d >= n.

## Installation

Install ILAMM from github:

```{r gh-installation, eval = FALSE}
install.packages("devtools")
devtools::install_github("XiaoouPan/ILAMM")
library(ILAMM)
```

## Getting help

Help on the functions can be accessed by typing "?", followed by function name at the R command prompt. 

## Common error messages

The package `ILAMM` is implemented in `Rcpp`, and the following error messages might appear when you first install it:

* Error: "...could not find build tools necessary to build ILAMM": For Windows you need Rtools, for Mac OS X you need to install Command Line Tools for XCode. See (https://support.rstudio.com/hc/en-us/articles/200486498-Package-Development-Prerequisites). 

* Error: "library not found for -lgfortran/-lquadmath": It means your gfortran binaries are out of date. This is a common environment specific issue. 

    1. In R 3.0.0 - R 3.3.0: Upgrading to R 3.4 is strongly recommended. Then go to the next step. Alternatively, you can try the instructions here: http://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/. 

    2. For >= R 3.4.* : download the installer from the here: https://gcc.gnu.org/wiki/GFortranBinaries#MacOS. Then run the installer.


## Functions

There are four functions, all of which are implemented by I-LAMM algorithm. 

* `ncvxReg`: Non-convex regularized regression (Lasso, SCAD, MCP). 
* `ncvxHuberReg`: Non-convex regularized Huber regression (Huber-Lasso, Huber-SCAD, Huber-MCP).
* `cvNcvxReg`: K-fold cross validation for non-convex regularized regression.
* `cvNcvxHuberReg`: K-fold cross validation for non-convex regularized Huber regression.

## Simple examples 

Here we generate data from a sparse linear model Y = X * beta + epsilon, where beta is sparse and epsilon consists of indepedent coordinates drawn from a log-normal distribution, which is asymmetrix and heavy-tailed. 

```{r}
library(ILAMM)
n = 50
d = 100
set.seed(2018)
X = matrix(rnorm(n * d), n, d)
beta = c(rep(2, 3), rep(0, d - 3))
Y = X %*% beta + rlnorm(n, 0, 1.2) - exp(1.2^2 / 2)
```

Then we fit five methods on {X, Y}: Lasso, SCAD, Huber-SCAD, MCP and Huber-MCP, and we can evidently find the advantages of Huber-SCAD and Huber-MCP over theor least square counterparts (SCAD and MCP).

```{r}
fitLasso = cvNcvxReg(X, Y, penalty = "Lasso")
betaLasso = fitLasso$beta
fitSCAD = cvNcvxReg(X, Y, penalty = "SCAD")
betaSCAD = fitSCAD$beta
fitHuberSCAD = cvNcvxHuberReg(X, Y, penalty = "SCAD")
betaHuberSCAD = fitHuberSCAD$beta
fitMCP = cvNcvxReg(X, Y, penalty = "MCP")
betaMCP = fitMCP$beta
fitHuberMCP = cvNcvxHuberReg(X, Y, penalty = "MCP")
betaHuberMCP = fitHuberMCP$beta
```

## Notes 

Function `cvNcvxHuberReg` might be slow, because we'll do a two-dimensional grid search for cross validation to determine the values of lambda and tau.

## Reference

Fan, J. and Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. J. Amer. Statist. Assoc. 96 1348–1360.

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841.

Huber, P. J. (1964). Robust estimation of a location parameter. Ann. Math. Statist. 35 73–101.

Sanderson, C. and Curtin, R. (2016). Armadillo: a template-based C++ library for linear algebra. J. Open. Src. Softw. 1 26.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Stat. Soc. Ser. B. Stat. Methodol. 58 267–288.

Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. Ann. Statist. 38 894–942.
