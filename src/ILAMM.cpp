# include <RcppArmadillo.h>
# include <cmath>
# include <string>
// [[Rcpp::depends(RcppArmadillo)]]

int sgn(const double x) {
  return (x > 0) - (x < 0);
}

arma::vec softThresh(const arma::vec& x, const arma::vec& lambda) {
  return arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros(x.size()));
}

arma::vec cmptLambda(const arma::vec& beta, const double lambda, const std::string penalty) {
  arma::vec rst = arma::zeros(beta.size());
  if (penalty == "Lasso") {
    rst = lambda * arma::ones(beta.size());
    rst(0) = 0;
  } else if (penalty == "SCAD") {
    double a = 3.7;
    double abBeta;
    for (int i = 1; i < (int)beta.size(); i++) {
      abBeta = std::abs(beta(i));
      if (abBeta <= lambda) {
        rst(i) = lambda;
      } else if (abBeta <= a * lambda) {
        rst(i) = (a * lambda - abBeta) / (a - 1);
      }
    }
  } else if (penalty == "MCP") {
    double a = 3;
    double abBeta;
    for (int i = 1; i < (int)beta.size(); i++) {
      abBeta = std::abs(beta(i));
      if (abBeta <= a * lambda) {
        rst(i) = lambda - abBeta / a;
      }
    }
  }
  return rst;
}

double loss(const arma::vec& Y, const arma::vec& Ynew, const std::string lossType,
            const double tau) {
  double rst = 0;
  if (lossType == "l2") {
    rst = arma::mean(arma::square(Y - Ynew)) / 2;
  } else if (lossType == "Huber") {
    arma::vec res = Y - Ynew;
    for (int i = 0; i < (int)Y.size(); i++) {
      if (std::abs(res(i)) <= tau) {
        rst += res(i) * res(i) / 2;
      } else {
        rst += tau * std::abs(res(i)) - tau * tau / 2;
      }
    }
    rst /= Y.size();
  }
  return rst;
}

arma::vec gradLoss(const arma::mat& X, const arma::vec& Y, const arma::vec& beta,
                   const std::string lossType, const double tau, const bool interecept) {
  arma::vec res = Y - X * beta;
  arma::vec rst = arma::zeros(beta.size());
  if (lossType == "l2") {
    rst = -1 * (res.t() * X).t();
  } else if (lossType == "Huber") {
    for (int i = 0; i < (int)Y.size(); i++) {
      if (std::abs(res(i)) <= tau) {
        rst -= res(i) * X.row(i).t();
      } else {
        rst -= tau * sgn(res(i)) * X.row(i).t();
      }
    }
  }
  if (!interecept) {
    rst(0) = 0;
  }
  return rst / Y.size();
}

arma::vec updateBeta(const arma::mat& X, const arma::vec& Y, arma::vec beta, const double phi,
                     const arma::vec& Lambda, const std::string lossType, const double tau,
                     const bool intercept) {
  arma::vec first = beta - gradLoss(X, Y, beta, lossType, tau, intercept) / phi;
  arma::vec second = Lambda / phi;
  return softThresh(first, second);
}

double cmptPsi(const arma::mat& X, const arma::vec& Y, const arma::vec& betaNew,
               const arma::vec& beta, const double phi, const std::string lossType,
               const double tau, const bool intercept) {
  arma::vec diff = betaNew - beta;
  double rst = loss(Y, X * beta, lossType, tau)
    + arma::as_scalar((gradLoss(X, Y, beta, lossType, tau, intercept)).t() * diff)
    + phi * arma::as_scalar(diff.t() * diff) / 2;
  return rst;
}

Rcpp::List LAMM(const arma::mat& X, const arma::vec& Y, const arma::vec& Lambda, arma::vec beta,
                const double phi, const std::string lossType, const double tau,
                const double gamma, const bool interecept) {
  double phiNew = phi;
  arma::vec betaNew = arma::vec();
  double FVal;
  double PsiVal;
  while (true) {
    betaNew = updateBeta(X, Y, beta, phiNew, Lambda, lossType, tau, interecept);
    FVal = loss(Y, X * betaNew, lossType, tau);
    PsiVal = cmptPsi(X, Y, betaNew, beta, phiNew, lossType, tau, interecept);
    if (FVal <= PsiVal) {
      break;
    }
    phiNew *= gamma;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phiNew);
}

//' The function fits (high-dimensional) regularized regression with non-convex penalties: Lasso, SCAD and MCP, and it's implemented via I-LAMM algorithm.
//'
//' The observed data are \eqn{(Y, X)}, where \eqn{Y} is an \eqn{n}-dimensional response vector and \eqn{X} is an \eqn{n} by \eqn{d} design matrix. We assume that \eqn{Y} depends on \eqn{X} through a linear model \eqn{Y = X \beta + \epsilon}, where \eqn{\epsilon} is an \eqn{n}-dimensional noise vector whose distribution can be asymmetrix and/or heavy-tailed. The design matrix \eqn{X} can be either high-dimensional or low-dimensional. Tunning parameter \eqn{\lambda} has a default setting but it can be user-specified. All the arguments except for \eqn{X} and \eqn{Y} have default settings.
//'
//' @title Non-convex regularized regression
//' @param X An \eqn{n} by \eqn{d} design matrix with each row being a sample and each column being a variable, either low-dimensional data (\eqn{d \le n}) or high-dimensional data (\eqn{d > n}) are allowed.
//' @param Y A continuous response vector with length \eqn{n}.
//' @param lambda Tuning parameter of regularized regression, its specified value should be positive. The default value is determined in this way: define \eqn{\lambda_max = max(|Y^T X|) / n}, and \eqn{\lambda_min = 0.01 * \lambda_max}, then \eqn{\lambda = exp(0.7 * log(\lambda_max) + 0.3 * log(\lambda_min))}.
//' @param penalty Type of non-convex penalties with default setting "SCAD", possible choices are: "Lasso", "SCAD" and "MCP".
//' @param phi0 The initial value of the isotropic parameter \eqn{\phi} in I-LAMM algorithm. The defalut value is 0.001.
//' @param gamma The inflation parameter in I-LAMM algorithm, in each iteration of I-LAMM, we will inflate \eqn{\phi} by \eqn{\gamma}. The defalut value is 1.5.
//' @param epsilon_c The tolerance level for contraction stage, iteration of contraction will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_c}. The defalut value is 1e-4.
//' @param epsilon_t The tolerance level for tightening stage, iteration of tightening will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_t}. The defalut value is 1e-4.
//' @param iteMax The maximal number of iteration in either contraction or tightening stage, if this number is reached, the convergence of I-LAMM is failed. The defalut value is 500.
//' @param intercept Boolean value indicating whether an intercept term should be included into the model. The default setting is \code{FALSE}.
//' @param itcpIncluded Boolean value indicating whether a column of 1's has been included in the design matrix \eqn{X}. The default setting is \code{FALSE}.
//' @return A list including the following terms will be returned:
//' \itemize{
//' \item \code{beta} The estimated \eqn{\beta}, a vector with length d + 1, with the first one being the value of intercept (0 if \code{intercept = FALSE}).
//' \item \code{phi} The final value of the isotropic parameter \eqn{\phi} in the last iteration of I-LAMM algorithm.
//' \item \code{penalty} The type of penalty.
//' \item \code{lambda} The value of \eqn{\lambda}.
//' \item \code{IteTightening} The number of tightenings in I-LAMM algorithm, and it's 0 if \code{penalty = "Lasso"}.
//' }
//' @author Xiaoou Pan, Qiang Sun, Wen-Xin Zhou
//' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841.
//' @seealso \code{\link{cvNcvxReg}}
//' @examples
//' n = 50
//' d = 100
//' set.seed(2018)
//' X = matrix(rnorm(n * d), n, d)
//' beta = c(rep(2, 3), rep(0, d - 3))
//' Y = X %*% beta + rnorm(n)
//' # Fit SCAD without intercept
//' fit = ncvxReg(X, Y)
//' fit$beta
//' # Fit MCP with intercept
//' fit = ncvxReg(X, Y, penalty = "MCP", intercept = TRUE)
//' fit$beta
//' @export
// [[Rcpp::export]]
Rcpp::List ncvxReg(arma::mat X, const arma::vec& Y, double lambda = -1,
                   std::string penalty = "SCAD", const double phi0 = 0.001,
                   const double gamma = 1.5, const double epsilon_c = 0.001,
                   const double epsilon_t = 0.001, const int iteMax = 500,
                   const bool intercept = false, const bool itcpIncluded = false) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  if (lambda <= 0) {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambda = std::exp((long double)(0.7 * std::log((long double)lambdaMax)
                                      + 0.3 * std::log((long double)lambdaMin)));
  }
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  // Contraction
  arma::vec Lambda = cmptLambda(beta, lambda, penalty);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = LAMM(X, Y, Lambda, beta, phi, "l2", 1, gamma, intercept);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  if (penalty != "Lasso") {
    arma::vec beta0 = arma::zeros(d + 1);
    while (iteT <= iteMax) {
      iteT++;
      beta = betaNew;
      beta0 = betaNew;
      Lambda = cmptLambda(beta, lambda, penalty);
      phi = phi0;
      ite = 0;
      while (ite <= iteMax) {
        ite++;
        listLAMM  = LAMM(X, Y, Lambda, beta, phi, "l2", 1, gamma, intercept);
        betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
        phi = listLAMM["phi"];
        phi = std::max(phi0, phi / gamma);
        if (arma::norm(betaNew - beta, "inf") <= epsilon_t) {
          break;
        }
        beta = betaNew;
      }
      if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
        break;
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi,
                            Rcpp::Named("penalty") = penalty, Rcpp::Named("lambda") = lambda,
                            Rcpp::Named("IteTightening") = iteT);
}

//' The function fits (high-dimensional) Huber regularized regression with non-convex penalties: Lasso, SCAD and MCP, and it's implemented via I-LAMM algorithm.
//'
//' The observed data are \eqn{(Y, X)}, where \eqn{Y} is an \eqn{n}-dimensional response vector and \eqn{X} is an \eqn{n} by \eqn{d} design matrix. We assume that \eqn{Y} depends on \eqn{X} through a linear model \eqn{Y = X \beta + \epsilon}, where \eqn{\epsilon} is an \eqn{n}-dimensional noise vector whose distribution can be asymmetrix and/or heavy-tailed. The design matrix \eqn{X} can be either high-dimensional or low-dimensional. Tunning parameters \eqn{\lambda} and \eqn{\tau} have default settings but they can be user-specified. All the arguments except for \eqn{X} and \eqn{Y} have default settings.
//'
//' @title Non-convex regularized Huber regression
//' @param X An \eqn{n} by \eqn{d} design matrix with each row being a sample and each column being a variable, either low-dimensional data (\eqn{d \le n}) or high-dimensional data (\eqn{d > n}) are allowed.
//' @param Y A continuous response vector with length \eqn{n}.
//' @param lambda Tuning parameter of regularized regression, its specified value should be positive. The default value is determined in this way: define \eqn{\lambda_max = max(|Y^T X|) / n}, and \eqn{\lambda_min = 0.01 * \lambda_max}, then \eqn{\lambda = exp(0.7 * log(\lambda_max) + 0.3 * log(\lambda_min))}.
//' @param penalty Type of non-convex penalties with default setting "SCAD", possible choices are: "Lasso", "SCAD" and "MCP".
//' @param tau Robustness parameter of Huber loss function, its specified value should be positive. The default value is determined in this way: define \eqn{R} as the residual from Lasso by fitting \code{ncvxReg} with \code{lambda}, and \eqn{\sigma_MAD = median(|R - median(R)|) / \Phi^(-1)(3/4)} is the median absolute deviation estimator, then \eqn{\tau = \sigma_MAD \sqrt(n / log(nd))}.
//' @param phi0 The initial value of the isotropic parameter \eqn{\phi} in I-LAMM algorithm. The defalut value is 0.001.
//' @param gamma The inflation parameter in I-LAMM algorithm, in each iteration of I-LAMM, we will inflate \eqn{\phi} by \eqn{\gamma}. The defalut value is 1.5.
//' @param epsilon_c The tolerance level for contraction stage, iteration of contraction will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_c}. The defalut value is 1e-4.
//' @param epsilon_t The tolerance level for tightening stage, iteration of tightening will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_t}. The defalut value is 1e-4.
//' @param iteMax The maximal number of iteration in either contraction or tightening stage, if this number is reached, the convergence of I-LAMM is failed. The defalut value is 500.
//' @param intercept Boolean value indicating whether an intercept term should be included into the model. The default setting is \code{FALSE}.
//' @param itcpIncluded Boolean value indicating whether a column of 1's has been included in the design matrix \eqn{X}. The default setting is \code{FALSE}.
//' @return A list including the following terms will be returned:
//' \itemize{
//' \item \code{beta} The estimated \eqn{\beta}, a vector with length d + 1, with the first one being the value of intercept (0 if \code{intercept = FALSE}).
//' \item \code{phi} The final value of the isotropic parameter \eqn{\phi} in the last iteration of I-LAMM algorithm.
//' \item \code{penalty} The type of penalty.
//' \item \code{lambda} The value of \eqn{\lambda}.
//' \item \code{tau} The value of \eqn{\tau}.
//' \item \code{IteTightening} The number of tightenings in I-LAMM algorithm, and it's 0 if \code{penalty = "Lasso"}.
//' }
//' @author Xiaoou Pan, Qiang Sun, Wen-Xin Zhou
//' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841.
//' @references Wang, L., Zheng, C., Zhou, W. and Zhou, W.-X. (2018). A New Principle for Tuning-Free Huber Regression. Preprint.
//' @seealso \code{\link{cvNcvxHuberReg}}
//' @examples
//' n = 50
//' d = 100
//' set.seed(2018)
//' X = matrix(rnorm(n * d), n, d)
//' beta = c(rep(2, 3), rep(0, d - 3))
//' Y = X %*% beta + rlnorm(n, 0, 1.2) - exp(1.2^2 / 2)
//' # Fit Huber-SCAD without intercept
//' fit = ncvxHuberReg(X, Y)
//' fit$beta
//' # Fit Huber-MCP with intercept
//' fit = ncvxHuberReg(X, Y, penalty = "MCP", intercept = TRUE)
//' fit$beta
//' @export
// [[Rcpp::export]]
Rcpp::List ncvxHuberReg(arma::mat X, const arma::vec& Y, double lambda = -1,
                std::string penalty = "SCAD", double tau = -1, const double phi0 = 0.001,
                const double gamma = 1.5, const double epsilon_c = 0.001,
                const double epsilon_t = 0.001, const int iteMax = 500,
                const bool intercept = false, const bool itcpIncluded = false, 
                const bool tf = false, const double constTau = 2) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  if (lambda <= 0) {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambda = std::exp((long double)(0.7 * std::log((long double)lambdaMax)
                      + 0.3 * std::log((long double)lambdaMin)));
  }
  if (!tf) {
    if (tau <= 0) {
      Rcpp::List listILAMM = ncvxReg(X, Y, lambda, "Lasso", phi0, gamma, epsilon_c, epsilon_t,
                                     iteMax, intercept, true);
      arma::vec betaLasso = Rcpp::as<arma::vec>(listILAMM["beta"]);
      arma::vec res = Y - X * betaLasso;
      double mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
      tau = mad * std::sqrt((long double)(n / std::log(n * d)));
    }
    arma::vec beta = arma::zeros(d + 1);
    arma::vec betaNew = arma::zeros(d + 1);
    // Contraction
    arma::vec Lambda = cmptLambda(beta, lambda, penalty);
    double phi = phi0;
    int ite = 0;
    Rcpp::List listLAMM;
    while (ite <= iteMax) {
      ite++;
      listLAMM = LAMM(X, Y, Lambda, beta, phi, "Huber", tau, gamma, intercept);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
    }
    int iteT = 0;
    // Tightening
    if (penalty != "Lasso") {
      arma::vec beta0 = arma::zeros(d + 1);
      while (iteT <= iteMax) {
        iteT++;
        beta = betaNew;
        beta0 = betaNew;
        Lambda = cmptLambda(beta, lambda, penalty);
        phi = phi0;
        ite = 0;
        while (ite <= iteMax) {
          ite++;
          listLAMM  = LAMM(X, Y, Lambda, beta, phi, "Huber", tau, gamma, intercept);
          betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
          phi = listLAMM["phi"];
          phi = std::max(phi0, phi / gamma);
          if (arma::norm(betaNew - beta, "inf") <= epsilon_t) {
            break;
          }
          beta = betaNew;
        }
        if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
          break;
        }
      }
    }
    return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi,
                              Rcpp::Named("penalty") = penalty, Rcpp::Named("lambda") = lambda,
                              Rcpp::Named("tau") = tau, Rcpp::Named("IteTightening") = iteT);
  } else {
    arma::vec res(n);
    double mad;
    if (tau <= 0) {
      Rcpp::List listILAMM = ncvxReg(X, Y, lambda, "Lasso", phi0, gamma, epsilon_c, epsilon_t,
                                     iteMax, intercept, true);
      arma::vec betaLasso = Rcpp::as<arma::vec>(listILAMM["beta"]);
      res = Y - X * betaLasso;
      mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
      tau = constTau * mad;
    }
    arma::vec beta = arma::zeros(d + 1);
    arma::vec betaNew = arma::zeros(d + 1);
    // Contraction
    arma::vec Lambda = cmptLambda(beta, lambda, penalty);
    double phi = phi0;
    int ite = 0;
    Rcpp::List listLAMM;
    while (ite <= iteMax) {
      ite++;
      listLAMM = LAMM(X, Y, Lambda, beta, phi, "Huber", tau, gamma, intercept);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
      res = Y - X * beta;
      mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
      tau = constTau * mad;
    }
    int iteT = 0;
    // Tightening
    if (penalty != "Lasso") {
      arma::vec beta0 = arma::zeros(d + 1);
      while (iteT <= iteMax) {
        iteT++;
        beta = betaNew;
        beta0 = betaNew;
        Lambda = cmptLambda(beta, lambda, penalty);
        phi = phi0;
        ite = 0;
        while (ite <= iteMax) {
          ite++;
          listLAMM  = LAMM(X, Y, Lambda, beta, phi, "Huber", tau, gamma, intercept);
          betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
          phi = listLAMM["phi"];
          phi = std::max(phi0, phi / gamma);
          if (arma::norm(betaNew - beta, "inf") <= epsilon_t) {
            break;
          }
          beta = betaNew;
          res = Y - X * beta;
          mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
          tau = constTau * mad;
        }
        if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
          break;
        }
      }
    }
    return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi,
                              Rcpp::Named("penalty") = penalty, Rcpp::Named("lambda") = lambda,
                              Rcpp::Named("tau") = tau, Rcpp::Named("IteTightening") = iteT);
  }
}

arma::uvec getIndex(const int n, const int low, const int up) {
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq >= low && seq <= up);
}

arma::uvec getIndexComp(const int n, const int low, const int up) {
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq < low || seq > up);
}

arma::vec tauConst(int n) {
  int end = n >> 1;
  int start = (n == end << 1) ? (end - 1) : end;
  arma::vec rst = arma::vec(n);
  int j = 0;
  for (int i = start; i > 0; i--) {
    rst(j++) = (double)1 / (1 << i);
  }
  for (int i = 0; i <= end; i++) {
    rst(j++) = 1 << i;
  }
  return rst;
}

//' The function performs k-fold cross validation for (high-dimensional) regularized regression with non-convex penalties: Lasso, SCAD and MCP, and it's implemented via I-LAMM algorithm.
//'
//' The observed data are \eqn{(Y, X)}, where \eqn{Y} is an \eqn{n}-dimensional response vector and \eqn{X} is an \eqn{n} by \eqn{d} design matrix. We assume that \eqn{Y} depends on \eqn{X} through a linear model \eqn{Y = X \beta + \epsilon}, where \eqn{\epsilon} is an \eqn{n}-dimensional noise vector whose distribution can be asymmetrix and/or heavy-tailed. The design matrix \eqn{X} can be either high-dimensional or low-dimensional. The sequence of \eqn{\lambda}'s has a default setting but it can be user-specified. All the arguments except for \eqn{X} and \eqn{Y} have default settings.
//'
//' @title K-fold cross validation for non-convex regularized regression
//' @param X An \eqn{n} by \eqn{d} design matrix with each row being a sample and each column being a variable, either low-dimensional data (\eqn{d \le n}) or high-dimensional data (\eqn{d > n}) are allowed.
//' @param Y A continuous response vector with length \eqn{n}.
//' @param lSeq Sequence of tuning parameter of regularized regression \eqn{\lambda}, every element should be positive. If it's not specified, the default sequence is generated in this way: define \eqn{\lambda_max = max(|Y^T X|) / n}, and \eqn{\lambda_min = 0.01 * \lambda_max}, then \code{lseq} is a sequence from \eqn{\lambda_max} to \eqn{\lambda_min} that decreases uniformly on log scale.
//' @param nlambda Number of \eqn{\lambda} to generate the default sequence \code{lSeq}. It's not necessary if \code{lSeq} is specified. The default value is 30.
//' @param penalty Type of non-convex penalties with default setting "SCAD", possible choices are: "Lasso", "SCAD" and "MCP".
//' @param phi0 The initial value of the isotropic parameter \eqn{\phi} in I-LAMM algorithm. The defalut value is 0.001.
//' @param gamma The inflation parameter in I-LAMM algorithm, in each iteration of I-LAMM, we will inflate \eqn{\phi} by \eqn{\gamma}. The defalut value is 1.5.
//' @param epsilon_c The tolerance level for contraction stage, iteration of contraction will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_c}. The defalut value is 1e-4.
//' @param epsilon_t The tolerance level for tightening stage, iteration of tightening will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_t}. The defalut value is 1e-4.
//' @param iteMax The maximal number of iteration in either contraction or tightening stage, if this number is reached, the convergence of I-LAMM is failed. The defalut value is 500.
//' @param nfolds The number of folds to conduct cross validation, values that are greater than 10 are not recommended, and it'll be modified to 10 if the input is greater than 10. The default value is 3.
//' @param intercept Boolean value indicating whether an intercept term should be included into the model. The default setting is \code{FALSE}.
//' @param itcpIncluded Boolean value indicating whether a column of 1's has been included in the design matrix \eqn{X}. The default setting is \code{FALSE}.
//' @return A list including the following terms will be returned:
//' \itemize{
//' \item \code{beta} The estimated \eqn{\beta} with \eqn{\lambda} determined by cross validation, it's a vector with length d + 1, with the first one being the value of intercept (0 if \code{intercept = FALSE}).
//' \item \code{penalty} The type of penalty.
//' \item \code{lambdaSeq} The sequence of \eqn{\lambda}'s for cross validation.
//' \item \code{mse} The mean squared error from cross validation, it's a vector with length \code{nlambda}.
//' \item \code{lambdaMin} The value of \eqn{\lambda} in \code{lambdaSeq} that minimized \code{mse}.
//' \item \code{nfolds} The number of folds for cross validation.
//' }
//' @author Xiaoou Pan, Qiang Sun, Wen-Xin Zhou
//' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841.
//' @seealso \code{\link{ncvxReg}}
//' @examples
//' n = 50
//' d = 100
//' set.seed(2018)
//' X = matrix(rnorm(n * d), n, d)
//' beta = c(rep(2, 3), rep(0, d - 3))
//' Y = X %*% beta + rnorm(n)
//' # Fit SCAD without intercept, with lambda determined by 3-folds cross validation
//' fit = cvNcvxReg(X, Y)
//' fit$beta
//' fit$lambdaMin
//' # Fit MCP with intercept, with lambda determined by 5-folds cross validation
//' fit = cvNcvxReg(X, Y, penalty = "MCP", intercept = TRUE, nfolds = 5)
//' fit$beta
//' fit$lambdaMin
//' @export
// [[Rcpp::export]]
Rcpp::List cvNcvxReg(arma::mat& X, const arma::vec& Y,
                    Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, int nlambda = 30,
                    const std::string penalty = "SCAD", const double phi0 = 0.001,
                    const double gamma = 1.5, const double epsilon_c = 0.001,
                    const double epsilon_t = 0.001, const int iteMax = 500, int nfolds = 3,
                    const bool intercept = false, const bool itcpIncluded = false) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = exp(arma::linspace(std::log((long double)lambdaMin),
                                   std::log((long double)lambdaMax), nlambda));
  }
  if (nfolds > 10 || nfolds > n) {
    nfolds = n < 10 ? n : 10;
  }
  int size = n / nfolds;
  arma::vec YPred(n);
  arma::vec betaHat(X.n_cols);
  arma::vec mse(nlambda);
  int low, up;
  arma::uvec idx, idxComp;
  Rcpp::List listILAMM;
  for (int i = 0; i < nlambda; i++) {
    for (int j = 0; j < nfolds; j++) {
      low = j * size;
      up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
      idx = getIndex(n, low, up);
      idxComp = getIndexComp(n, low, up);
      listILAMM = ncvxReg(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i), penalty,
                                     phi0, gamma, epsilon_c, epsilon_t, iteMax, intercept, true);
      betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
      YPred.rows(idx) = X.rows(idx) * betaHat;
    }
    mse(i) = arma::norm(Y - YPred, 2);
  }
  arma::uword cvIdx = mse.index_min();
  listILAMM = ncvxReg(X, Y, lambdaSeq(cvIdx), penalty, phi0, gamma, epsilon_c,
                                 epsilon_t, iteMax, intercept, true);
  arma::vec beta = Rcpp::as<arma::vec>(listILAMM["beta"]);
  return Rcpp::List::create(Rcpp::Named("beta") = beta, Rcpp::Named("penalty") = penalty,
                            Rcpp::Named("lambdaSeq") = lambdaSeq, Rcpp::Named("mse") = mse,
                            Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), Rcpp::Named("nfolds") = nfolds);
}

double f1(const double x, const arma::vec& resSq, const int n) {
  return arma::sum(arma::min(resSq, x * arma::ones(n))) / (n * x) - std::log(n) / n;
}

double rootf1(const arma::vec& resSq, const int n, double low, double up, 
              const double tol = 0.00001, const int maxIte = 500) {
  int ite = 0;
  double mid;
  double val;
  while (ite <= maxIte && up - low > tol) {
    mid = (up + low) / 2;
    val = f1(mid, resSq, n);
    if (val == 0) {
      return mid;
    } else if (val < 0) {
      up = mid;
    } else {
      low = mid;
    }
    ite++;
  }
  return (low + up) / 2;
}

double huberMean(const arma::vec& X, const double epsilon = 0.00001, const int iteMax = 500) {
  int n = X.size();
  double muOld = 0;
  double muNew = arma::mean(X);
  double tauOld = 0;
  double tauNew = arma::stddev(X) * std::sqrt((long double)n / std::log(n));
  int iteNum = 0;
  arma::vec res(n);
  arma::vec resSq(n);
  arma::vec w(n);
  while (((std::abs(muNew - muOld) > epsilon) || (std::abs(tauNew - tauOld) > epsilon)) && iteNum < iteMax) {
    muOld = muNew;
    tauOld = tauNew;
    res = X - muOld * arma::ones(n);
    resSq = arma::square(res);
    tauNew = std::sqrt((long double)rootf1(resSq, n, arma::min(resSq), arma::sum(resSq)));
    w = arma::min(tauNew * arma::ones(n) / arma::abs(res), arma::ones(n));
    muNew = arma::as_scalar(X.t() * w) / arma::sum(w);
    iteNum++;
  }
  return muNew;
}

double pairPred(const arma::mat& X, const arma::vec& Y, const arma::vec& beta) {
  int n = X.n_rows;
  int d = X.n_cols - 1;
  int m = n * (n - 1) >> 1;
  arma::mat pairX(m, d + 1);
  arma::vec pairY(m);
  int k = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      pairX.row(k) = X.row(i) - X.row(j);
      pairY(k++) = Y(i) - Y(j);
    }
  }
  arma::vec predY = pairX * beta;
  return arma::sum(arma::square(pairY - predY));
}

//' The function performs k-fold cross validation for (high-dimensional) Huber regularized regression with non-convex penalties: Lasso, SCAD and MCP, and it's implemented via I-LAMM algorithm.
//'
//' The observed data are \eqn{(Y, X)}, where \eqn{Y} is an \eqn{n}-dimensional response vector and \eqn{X} is an \eqn{n} by \eqn{d} design matrix. We assume that \eqn{Y} depends on \eqn{X} through a linear model \eqn{Y = X \beta + \epsilon}, where \eqn{\epsilon} is an \eqn{n}-dimensional noise vector whose distribution can be asymmetrix and/or heavy-tailed. The design matrix \eqn{X} can be either high-dimensional or low-dimensional. The sequence of \eqn{\lambda}'s and \eqn{\tau}'s have default settings but they can be user-specified. All the arguments except for \eqn{X} and \eqn{Y} have default settings.
//'
//' @title K-fold cross validation for non-convex regularized Huber regression
//' @param X An \eqn{n} by \eqn{d} design matrix with each row being a sample and each column being a variable, either low-dimensional data (\eqn{d \le n}) or high-dimensional data (\eqn{d > n}) are allowed.
//' @param Y A continuous response vector with length \eqn{n}.
//' @param lSeq Sequence of tuning parameter of regularized regression \eqn{\lambda}, every element should be positive. If it's not specified, the default sequence is generated in this way: define \eqn{\lambda_max = max(|Y^T X|) / n}, and \eqn{\lambda_min = 0.01 * \lambda_max}, then \code{lseq} is a sequence from \eqn{\lambda_max} to \eqn{\lambda_min} that decreases uniformly on log scale.
//' @param nlambda Number of \eqn{\lambda} to generate the default sequence \code{lSeq}. It's not necessary if \code{lSeq} is specified. The default value is 30.
//' @param penalty Type of non-convex penalties with default setting "SCAD", possible choices are: "Lasso", "SCAD" and "MCP".
//' @param tSeq Sequence of robustness parameter of Huber loss \eqn{\tau}, every element should be positive. If it's not specified, the default sequence is generated in this way: define \eqn{R} as the residual from Lasso by fitting \code{cvNcvxReg} with \code{lSeq}, and \eqn{\sigma_MAD = median(|R - median(R)|) / \Phi^(-1)(3/4)} is the median absolute deviation estimator, then \code{tSeq} = \eqn{2^j * \sigma_MAD \sqrt(n / log(nd))}, where \eqn{j} are integers from -\code{ntau}/2 to \code{ntau}/2.
//' @param ntau Number of \eqn{\tau} to generate the default sequence \code{tSeq}. It's not necessary if \code{tSeq} is specified. The default value is 5.
//' @param phi0 The initial value of the isotropic parameter \eqn{\phi} in I-LAMM algorithm. The defalut value is 0.001.
//' @param gamma The inflation parameter in I-LAMM algorithm, in each iteration of I-LAMM, we will inflate \eqn{\phi} by \eqn{\gamma}. The defalut value is 1.5.
//' @param epsilon_c The tolerance level for contraction stage, iteration of contraction will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_c}. The defalut value is 1e-4.
//' @param epsilon_t The tolerance level for tightening stage, iteration of tightening will stop when \eqn{||\beta_new - \beta_old||_2 / \sqrt(d + 1) < \epsilon_t}. The defalut value is 1e-4.
//' @param iteMax The maximal number of iteration in either contraction or tightening stage, if this number is reached, the convergence of I-LAMM is failed. The defalut value is 500.
//' @param nfolds The number of folds to conduct cross validation, values that are greater than 10 are not recommended, and it'll be modified to 10 if the input is greater than 10. The default value is 3.
//' @param intercept Boolean value indicating whether an intercept term should be included into the model. The default setting is \code{FALSE}.
//' @param itcpIncluded Boolean value indicating whether a column of 1's has been included in the design matrix \eqn{X}. The default setting is \code{FALSE}.
//' @return A list including the following terms will be returned:
//' \itemize{
//' \item \code{beta} The estimated \eqn{\beta} with \eqn{\lambda} and \eqn{\tau} determined by cross validation, it's a vector with length d + 1, with the first one being the value of intercept (0 if \code{intercept = FALSE}).
//' \item \code{penalty} The type of penalty.
//' \item \code{lambdaSeq} The sequence of \eqn{\lambda}'s for cross validation.
//' \item \code{tauSeq} The sequence of \eqn{\tau}'s for cross validation.
//' \item \code{mse} The mean squared error from cross validation, it's a matrix with dimension \code{nlambda} by \code{ntau}.
//' \item \code{lambdaMin} The value of \eqn{\lambda} in \code{lSeq} that minimized \code{mse}.
//' \item \code{tauMin} The value of \eqn{\tau} in \code{tSeq} that minimized \code{mse}.
//' \item \code{nfolds} The number of folds for cross validation.
//' }
//' @author Xiaoou Pan, Qiang Sun, Wen-Xin Zhou
//' @references Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist. 46 814–841.
//' @references Wang, L., Zheng, C., Zhou, W. and Zhou, W.-X. (2018). A New Principle for Tuning-Free Huber Regression. Preprint.
//' @seealso \code{\link{ncvxHuberReg}}
//' @examples
//' n = 50
//' d = 100
//' set.seed(2018)
//' X = matrix(rnorm(n * d), n, d)
//' beta = c(rep(2, 3), rep(0, d - 3))
//' Y = X %*% beta + rlnorm(n, 0, 1.2) - exp(1.2^2 / 2)
//' # Fit SCAD without intercept, with lambda and tau determined by 3-folds cross validation
//' fit = cvNcvxHuberReg(X, Y)
//' fit$beta
//' fit$lambdaMin
//' fit$tauMin
//' # Fit MCP with intercept, with lambda and tau determined by 3-folds cross validation
//' fit = cvNcvxHuberReg(X, Y, penalty = "MCP", intercept = TRUE)
//' fit$beta
//' fit$lambdaMin
//' fit$tauMin
//' @export
// [[Rcpp::export]]
Rcpp::List cvNcvxHuberReg(arma::mat& X, const arma::vec& Y,
                  Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, int nlambda = 30,
                  const std::string penalty = "SCAD",
                  Rcpp::Nullable<Rcpp::NumericVector> tSeq = R_NilValue, int ntau = 5,
                  const double phi0 = 0.001, const double gamma = 1.5,
                  const double epsilon_c = 0.001, const double epsilon_t = 0.001,
                  const int iteMax = 500, int nfolds = 3, const bool intercept = false,
                  const bool itcpIncluded = false, const bool tf = false, 
                  const double constTau = 2) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  if (nfolds > 10 || nfolds > n) {
    nfolds = n < 10 ? n : 10;
  }
  int size = n / nfolds;
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = exp(arma::linspace(std::log((long double)lambdaMin),
                    std::log((long double)lambdaMax), nlambda));
  }
  if (!tf) {
    arma::vec tauSeq = arma::vec();
    Rcpp::List listILAMM;
    if (tSeq.isNotNull()) {
      tauSeq = Rcpp::as<arma::vec>(tSeq);
      ntau = tauSeq.size();
    } else {
      listILAMM = cvNcvxReg(X, Y, lSeq, nlambda, "Lasso", phi0, gamma, epsilon_c,
                                       epsilon_t, iteMax, nfolds, intercept, true);
      arma::vec betaLasso = Rcpp::as<arma::vec>(listILAMM["beta"]);
      arma::vec res = Y - X * betaLasso;
      double mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
      arma::vec tauCon = tauConst(ntau);
      tauSeq = mad * std::sqrt((long double)(n / std::log(n * d))) * tauCon;
    }
    arma::vec YPred(n);
    arma::vec betaHat(d + 1);
    arma::mat mse(nlambda, ntau);
    int low, up;
    arma::uvec idx, idxComp;
    for (int i = 0; i < nlambda; i++) {
      for (int k = 0; k < ntau; k++) {
        for (int j = 0; j < nfolds; j++) {
          low = j * size;
          up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
          idx = getIndex(n, low, up);
          idxComp = getIndexComp(n, low, up);
          listILAMM = ncvxHuberReg(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i),
                                              penalty, tauSeq(k), phi0, gamma, epsilon_c, epsilon_t,
                                              iteMax, intercept, true, tf, constTau);
          betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
          YPred.rows(idx) = X.rows(idx) * betaHat;
        }
        mse(i, k) = arma::norm(Y - YPred, 2);
      }
    }
    arma::uword cvIdx = mse.index_min();
    arma::uword idxLambda = cvIdx - (cvIdx / nlambda) * nlambda;
    arma::uword idxTau = cvIdx / nlambda;
    listILAMM = ncvxHuberReg(X, Y, lambdaSeq(idxLambda), penalty, tauSeq(idxTau), phi0,
                                        gamma, epsilon_c, epsilon_t, iteMax, intercept, true, tf,
                                        constTau);
    arma::vec beta = Rcpp::as<arma::vec>(listILAMM["beta"]);
    return Rcpp::List::create(Rcpp::Named("beta") = beta, Rcpp::Named("penalty") = penalty,
                              Rcpp::Named("lambdaSeq") = lambdaSeq, Rcpp::Named("tauSeq") = tauSeq,
                              Rcpp::Named("mse") = mse, Rcpp::Named("lambdaMin") = lambdaSeq(idxLambda),
                              Rcpp::Named("tauMin") = tauSeq(idxTau), Rcpp::Named("nfolds") = nfolds);
  } else if (!intercept) {
    arma::vec YPred(n);
    arma::vec betaHat(d + 1);
    arma::vec mse(nlambda);
    int low, up;
    arma::uvec idx, idxComp;
    Rcpp::List listILAMM;
    for (int i = 0; i < nlambda; i++) {
      for (int j = 0; j < nfolds; j++) {
        low = j * size;
        up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
        idx = getIndex(n, low, up);
        idxComp = getIndexComp(n, low, up);
        listILAMM = ncvxHuberReg(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i),
                                 penalty, -1, phi0, gamma, epsilon_c, epsilon_t,
                                 iteMax, intercept, true, tf, constTau);
        betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
        YPred.rows(idx) = X.rows(idx) * betaHat;
      }
      mse(i) = arma::norm(Y - YPred, 2);
    }
    arma::uword cvIdx = mse.index_min();
    listILAMM = ncvxHuberReg(X, Y, lambdaSeq(cvIdx), penalty, -1, phi0, gamma, epsilon_c, 
                             epsilon_t, iteMax, intercept, true, tf, constTau);
    arma::vec beta = Rcpp::as<arma::vec>(listILAMM["beta"]);
    return Rcpp::List::create(Rcpp::Named("beta") = beta, Rcpp::Named("penalty") = penalty,
                              Rcpp::Named("lambdaSeq") = lambdaSeq, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx),
                              Rcpp::Named("tau") = listILAMM["tau"], Rcpp::Named("nfolds") = nfolds);
  } else {
    arma::vec betaHat(d + 1);
    arma::vec mse = arma::zeros(nlambda);
    int low, up;
    arma::uvec idx, idxComp;
    Rcpp::List listILAMM;
    for (int i = 0; i < nlambda; i++) {
      for (int j = 0; j < nfolds; j++) {
        low = j * size;
        up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
        idx = getIndex(n, low, up);
        idxComp = getIndexComp(n, low, up);
        listILAMM = ncvxHuberReg(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i),
                                            penalty, -1, phi0, gamma, epsilon_c, epsilon_t,
                                            iteMax, intercept, true, tf, constTau);
        betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
        mse(i) += pairPred(X.rows(idx), Y.rows(idx), betaHat);
      }
    }
    arma::uword cvIdx = mse.index_min();
    listILAMM = ncvxHuberReg(X, Y, lambdaSeq(cvIdx), penalty, -1, phi0, gamma, epsilon_c, 
                                        epsilon_t, iteMax, intercept, true, tf, constTau);
    arma::vec beta = Rcpp::as<arma::vec>(listILAMM["beta"]);
    beta(0) = huberMean(Y - X.cols(1, d) * beta.rows(1, d));
    return Rcpp::List::create(Rcpp::Named("beta") = beta, Rcpp::Named("penalty") = penalty,
                              Rcpp::Named("lambdaSeq") = lambdaSeq, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx),
                              Rcpp::Named("tau") = listILAMM["tau"], Rcpp::Named("nfolds") = nfolds);
  }
}
