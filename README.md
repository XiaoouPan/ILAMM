---
output: github_document

references:
- id: ilamm
  title: 'I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error.' 
  author:
  - family: Fan
    given: J.
  - family: Liu
    given: H.
  - family: Sun
    given: Q.
  - family: Zhang
    given: T.
  container-title: Annals of Statistics
  URL: 'https://projecteuclid.org/euclid.aos/1522742437'
  issued: 46 814–841
    year: 2018

<!-- README.md is generated from README.Rmd. Please edit that file -->

```{r, echo = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.path = "README-"
)
```

# ILAMM

Non-convex Regularized Robust Regression with I-LAMM Algorithm

## Goal of the package

The package implemented I-LAMM algorithm in C++ on non-convex regularized robust regression with penalties Lasso, smoothly clipped absolute deviation (SCAD) and minimax concave penalty (MCP). Tuning parameters $\lambda$ and $\tau$ (for Huber loss) can be determined by cross validation. As a by-product, the package can also run least squares counterparts (i.e., Lasso, SCAD and MCP). See the papers on this method, [@ilamm], for more details. 

The observed data $X$ is a $n \times d$ matrix, where both low-dimension ($d \le n$) and high-dimension ($d > n$) are allowed, response $Y$ is a continuous vector with length $n$. It's assumed that $Y$ come from the model $Y = X \beta + \epsilon$, where $\epsilon$ may come from asymmetrix and/or heavy-tailed distributions. 

## Installation

Install ILAMM from github with:

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


##Functions

There are four functions, all of which are implemented by I-LAMM algorithm. 

* `ncvxReg`: Non-convex regularized regression (Lasso, SCAD, MCP). 
* `ncvxHuberReg`: Non-convex regularized Huber regression (Huber-Lasso, Huber-SCAD, Huber-MCP).
* `cvNcvxReg`: K-fold cross validation for non-convex regularized regression.
* `cvNcvxHuberReg`: K-fold cross validation for non-convex regularized Huber regression.

## Simple examples 

Here we generate high-dimensional data and linear model $Y = X \beta + \epsilon$, where $\beta$ is sparse and $\epsilon$ are from log-normal distribution, so they are asymmetrix and heavy-tailed. 

```{r}
library(ILAMM)
n = 50
d = 100
set.seed(2018)
X = matrix(rnorm(n * d), n, d)
beta = c(rep(2, 3), rep(0, d - 3))
Y = X %*% beta + rlnorm(n, 0, 1.2) - exp(1.2^2 / 2)
```

Then we fit five methods on $\{X, Y\}$: Lasso, SCAD, Huber-SCAD, MCP and Huber-MCP, and we can evidently find the advantages of Huber-SCAD and Huber-MCP over theor least square counterparts (SCAD and MCP).

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

Function `cvNcvxHuberReg` might be slow, because we'll do a two-dimensional grid search for cross validation to determine the values of $\lambda$ and $\tau$.
