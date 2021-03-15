#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>
#include <fstream>

#include "nelson/MatrixWWtMultiplier.hpp"
#include "nelson/MatrixDenseWrapper.hpp"
#include "nelson/MatrixSparseWrapper.hpp"

constexpr int BR = 6;
constexpr int BC = 3;

constexpr int matUType = mat::BlockDiagonal;
constexpr int matWType = mat::BlockCoeffSparse;
constexpr int matSType = mat::BlockCoeffSparse;

int main(int argc, char* argv[]) {
  const Problem * problemPtr = &ProblemCollections::ladybug_49_7776;
  std::unique_ptr<Problem> newProblem;
  if (argc == 2) {
    newProblem.reset(new Problem());
    bool ok = newProblem->load(argv[1]);
    if (!ok) {
      std::cerr << "error reading " << argv[1] << std::endl;
      std::exit(-1);
    }
    else {
      problemPtr = newProblem.get();
    }
  }
  const Problem& problem = *problemPtr;

  typename mat::MatrixBlockIterableTypeTraits<matUType, double, mat::ColMajor, BR, BR, mat::Dynamic, mat::Dynamic>::MatrixType U;
  mat::SparsityPattern<mat::ColMajor>::SPtr spU(new mat::SparsityPattern<mat::ColMajor>(problem.nCameras, problem.nCameras));
  spU->setDiagonal();

  U.resize(mat::MatrixBlockDescriptor<BR, BR, mat::Dynamic, mat::Dynamic>(BR, problem.nCameras, BR, problem.nCameras), spU);
  U.setZero();
  for (int bi = 0; bi < U.nonZeroBlocks(); bi++) {
    U.blockByUID(bi).setRandom();
    //U.blockByUID(bi).setZero();
  }

  typename mat::MatrixBlockIterableTypeTraits<matWType, double, mat::RowMajor, BR, BC, mat::Dynamic, mat::Dynamic>::MatrixType W;
  mat::SparsityPattern<mat::RowMajor>::SPtr sp(new mat::SparsityPattern<mat::RowMajor>(problem.nCameras, problem.nPoints));
  for (int i = 0; i < problem.edges.size(); i++) {
    const auto& e = problem.edges[i];
    sp->add(e.first, e.second);
  }

  W.resize(mat::MatrixBlockDescriptor<BR, BC, mat::Dynamic, mat::Dynamic>(BR, problem.nCameras, BC, problem.nPoints), sp);
  W.setZero();
  for (int bi = 0; bi < W.nonZeroBlocks(); bi++) {
    W.blockByUID(bi).setRandom();
    //W.blockByUID(bi).setZero();
  }

  nelson::MatrixWWtMultiplier<matSType, double, BR, mat::Dynamic> mwwt;
  const int half_maxthreads = nelson::ParallelExecSettings::maxSupportedThreads() / 2;
  const int maxthreads = nelson::ParallelExecSettings::maxSupportedThreads();

  // pre, test if it is faster to count on sparsity pattern or on the matrix version
  {
    auto t0 = std::chrono::steady_clock::now();
    Eigen::Matrix<int, 1, Eigen::Dynamic> counts = Eigen::Matrix<int, 1, Eigen::Dynamic>::Zero(W.numBlocksCol());
    for (int r = 0; r < sp->outerSize(); r++) {
      for (auto in : sp->inner(r)) {
        counts(in)++;
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    double t = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Iterate on SP and count " << t << std::endl;

    t0 = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<int, Eigen::RowMajor> spMat = W.sparsityPattern().toSparseMatrix();
    auto tint = std::chrono::steady_clock::now();
    Eigen::Matrix<int, 1, Eigen::Dynamic> counts2 = Eigen::Matrix<int, 1, Eigen::Dynamic>::Zero(W.numBlocksCol());
    for (int r = 0; r < spMat.outerSize(); ++r) {
      for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(spMat, r); it; ++it) {
        counts2(it.col())++;
      }
    }
    t1 = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "SP to mat and iterate to count " << t << std::endl;
    std::cout << " -- convert time " << std::chrono::duration<double>(tint - t0).count() << std::endl;
    std::cout << " -- count time " << std::chrono::duration<double>(t1 - tint).count() << std::endl;

    int err = (counts - counts2).cwiseAbs().maxCoeff();
    if (err != 0) std::cerr << "ERR! " << err << std::endl;
    assert(err == 0);
  }

  // with multiplier
  {
    auto t0 = std::chrono::steady_clock::now();
    mwwt.prepare(U, W);
    auto t1 = std::chrono::steady_clock::now();
    mwwt.settings().setSingleThread();
    mwwt.multiply(U, W, W);
    auto t2 = std::chrono::steady_clock::now();
    mwwt.settings().setNumThreads(half_maxthreads);
    mwwt.multiply(U, W, W);
    auto t3 = std::chrono::steady_clock::now();
    mwwt.settings().setNumThreadsMax();
    mwwt.multiply(U, W, W);
    auto t4 = std::chrono::steady_clock::now();

    double t_prep = std::chrono::duration<double>(t1 - t0).count();
    double t_mult = std::chrono::duration<double>(t2 - t1).count();
    double t_tot = std::chrono::duration<double>(t2 - t0).count();
    double t_mult2 = std::chrono::duration<double>(t3 - t2).count();
    double t_mult3 = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "t_prep " << t_prep << std::endl;
    std::cout << "t_mult " << t_mult << std::endl;
    std::cout << "t_tot  " << t_tot << std::endl;
    std::cout << "t_mult(" << half_maxthreads << ") " << t_mult2 << std::endl;
    std::cout << "t_mult(" << maxthreads << ") " << t_mult3 << std::endl;
  }

  // standard
  nelson::SparseWrapper<matUType, double, mat::ColMajor, BR, BR, mat::Dynamic, mat::Dynamic> U_Wrap;
  U_Wrap.set(&U);
  Eigen::SparseMatrix<double> WWt;
  {
    auto t0 = std::chrono::steady_clock::now();
    WWt = U_Wrap.mat() - W.mat() * W.mat().transpose();
    auto t1 = std::chrono::steady_clock::now();
    double t_tot = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "t_tot  " << t_tot << std::endl;

    auto t20 = std::chrono::steady_clock::now();
    WWt = U_Wrap.mat() - W.mat() * W.mat().transpose();
    auto t21 = std::chrono::steady_clock::now();
    double t_tot2 = std::chrono::duration<double>(t21 - t20).count();
    std::cout << "t_tot(2)  " << t_tot2 << std::endl;
    WWt = U_Wrap.mat() - W.mat() * W.mat().transpose();
    auto t22 = std::chrono::steady_clock::now();
    double t_tot3 = std::chrono::duration<double>(t22 - t21).count();
    std::cout << "t_tot(3)  " << t_tot3 << std::endl;

  }
 
  // checks 
  nelson::SparseWrapper<matSType, double, mat::ColMajor, BR, BR, mat::Dynamic, mat::Dynamic> WWt_Wrap;
  WWt_Wrap.set(&mwwt.result());

  Eigen::SparseMatrix<double> diff = (WWt_Wrap.mat() - WWt).triangularView<Eigen::Upper>();
  
  std::cout << "Eigen::NumTraits<double>::dummy_precision() " << Eigen::NumTraits<double>::dummy_precision() << std::endl;

  double maxErr = diff.coeffs().cwiseAbs().maxCoeff();
  bool ok = (diff.coeffs().cwiseAbs() < 16 * Eigen::NumTraits<double>::dummy_precision()).all();
  if(false) {
    std::ofstream f;
    f.open("loadWWt.m");
    f << "WWt = [" << std::endl;
    f << WWt_Wrap.mat() << std::endl;
    f << "];" << std::endl;
  }
  if (!ok) {
    std::cerr << "ERROR!! max is " << maxErr << std::endl;
    {
      std::ofstream f;
      f.open("loadDiffMat.m");
      f << "D = [" << std::endl;
      f << diff << std::endl;
      f << "];" << std::endl;
    }
    std::exit(-1);
  }
  else {
    std::cerr << "OK" << std::endl;
  }

  return 0;
}