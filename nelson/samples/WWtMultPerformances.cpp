#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>

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
  }

  nelson::MatrixWWtMultiplier<matSType, double, mat::ColMajor, BR, mat::Dynamic> mwwt;

  //mwwt.settings().setSingleThread();
  //mwwt.settings().setNumThreads(4);
  mwwt.settings().setNumThreadsMax();

  // with multiplier
  {
    auto t0 = std::chrono::steady_clock::now();
    mwwt.prepare(U, W);
    auto t1 = std::chrono::steady_clock::now();
    mwwt.multiply(U, W, W);
    auto t2 = std::chrono::steady_clock::now();
    mwwt.multiply(U, W, W);
    auto t3 = std::chrono::steady_clock::now();

    double t_prep = std::chrono::duration<double>(t1 - t0).count();
    double t_mult = std::chrono::duration<double>(t2 - t1).count();
    double t_tot = std::chrono::duration<double>(t2 - t0).count();
    double t_mult2 = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "t_prep " << t_prep << ", t_mult " << t_mult << std::endl;
    std::cout << "t_tot  " << t_tot << std::endl;
    std::cout << "t_mult(2) " << t_mult2 << std::endl;
  }

  // standard
  nelson::SparseWrapper<matUType, double, mat::ColMajor, BR, BR, mat::Dynamic, mat::Dynamic> U_Wrap;
  U_Wrap.set(&U);

  {
    auto t0 = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<double> WWt = U_Wrap.mat() - W.mat() * W.mat().transpose();
    auto t1 = std::chrono::steady_clock::now();
    double t_tot = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "t_tot  " << t_tot << std::endl;

    auto t20 = std::chrono::steady_clock::now();
    WWt = U_Wrap.mat() - W.mat() * W.mat().transpose();
    auto t21 = std::chrono::steady_clock::now();
    double t_tot2 = std::chrono::duration<double>(t21 - t20).count();
    std::cout << "t_tot(2)  " << t_tot2 << std::endl;

  }
 
  return 0;
}