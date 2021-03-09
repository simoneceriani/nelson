#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>
#include <fstream>

#include "nelson/MatrixWtXMultiplier.hpp"
#include "nelson/MatrixDenseWrapper.hpp"
#include "nelson/MatrixSparseWrapper.hpp"

constexpr int BR = 6;
constexpr int BC = 3;

template<int matWType> 
void testFunction(const Problem& problem) {

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

  mat::VectorBlock<double, BR, mat::Dynamic> xU;
  xU.resize(W.blockDescriptor().rowDescriptionCSPtr());
  xU.mat().setRandom();

  mat::VectorBlock<double, BC, mat::Dynamic> bV, bVOrig, bV2;
  bV.resize(W.blockDescriptor().colDescriptionCSPtr());
  bV2.resize(W.blockDescriptor().colDescriptionCSPtr());
  bV.mat().setRandom();
  bVOrig.mat() = bV.mat();
  bV2.mat() = bV.mat();


  nelson::MatrixWtXMultiplier<matWType, double, BR, BC, mat::Dynamic, mat::Dynamic> mwwt;
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

  // test prepare with shared pattern
  {
    nelson::MatrixWtXMultiplier<matWType, double, BR, BC, mat::Dynamic, mat::Dynamic> mwwt2;
    auto t0 = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<int, Eigen::RowMajor> spMat = W.sparsityPattern().toSparseMatrix();
    auto t1 = std::chrono::steady_clock::now();
    mwwt2.prepare(W, &spMat);
    auto t2 = std::chrono::steady_clock::now();
    double t_conv = std::chrono::duration<double>(t1 - t0).count();
    double t_prepare = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "****** shared sparsity pattern test ****" << std::endl;
    std::cout << "toSparseMatrix " << t_conv << std::endl;
    std::cout << "preparation " << t_prepare << std::endl;

  }

  // with multiplier
  {
    auto t0 = std::chrono::steady_clock::now();
    mwwt.prepare(W);
    auto t1 = std::chrono::steady_clock::now();
    mwwt.settings().setSingleThread();
    mwwt.rightMultVectorSub(W, xU, bV);
    auto t20 = std::chrono::steady_clock::now();
    bV.mat() = bVOrig.mat();
    auto t21 = std::chrono::steady_clock::now();
    mwwt.settings().setNumThreads(half_maxthreads);
    mwwt.rightMultVectorSub(W, xU, bV);
    auto t30 = std::chrono::steady_clock::now();
    bV.mat() = bVOrig.mat();
    auto t31 = std::chrono::steady_clock::now();
    mwwt.settings().setNumThreadsMax();
    mwwt.rightMultVectorSub(W, xU, bV);
    auto t4 = std::chrono::steady_clock::now();

    double t_prep = std::chrono::duration<double>(t1 - t0).count();
    double t_mult = std::chrono::duration<double>(t20 - t1).count();
    double t_tot = std::chrono::duration<double>(t20 - t0).count();
    double t_mult2 = std::chrono::duration<double>(t30 - t21).count();
    double t_mult3 = std::chrono::duration<double>(t4 - t31).count();

    std::cout << "t_prep " << t_prep << std::endl;
    std::cout << "t_mult " << t_mult << std::endl;
    std::cout << "t_tot  " << t_tot << std::endl;
    std::cout << "t_mult(" << half_maxthreads << ") " << t_mult2 << std::endl;
    std::cout << "t_mult(" << maxthreads << ") " << t_mult3 << std::endl;
  }

  // standard
  {
    auto t0c = std::chrono::steady_clock::now();
    nelson::SparseWrapper<matWType, double, mat::RowMajor, BR, BC, mat::Dynamic, mat::Dynamic> W_Wrap;
    W_Wrap.set(&W);
    auto t1c = std::chrono::steady_clock::now();
    double t_copy = std::chrono::duration<double>(t1c - t0c).count();
    std::cout << "t_copy (if any) to W_wrap  " << t_copy << std::endl;


    auto t0 = std::chrono::steady_clock::now();
    bV2.mat() = bVOrig.mat() - W_Wrap.mat().transpose() * xU.mat();
    auto t1 = std::chrono::steady_clock::now();
    double t_tot = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "t_tot  " << t_tot << std::endl;

    auto t20 = std::chrono::steady_clock::now();
    bV2.mat() = bVOrig.mat() - W_Wrap.mat().transpose() * xU.mat();
    auto t21 = std::chrono::steady_clock::now();
    double t_tot2 = std::chrono::duration<double>(t21 - t20).count();
    std::cout << "t_tot(2)  " << t_tot2 << std::endl;
    bV2.mat() = bVOrig.mat() - W_Wrap.mat().transpose() * xU.mat();
    auto t22 = std::chrono::steady_clock::now();
    double t_tot3 = std::chrono::duration<double>(t22 - t21).count();
    std::cout << "t_tot(3)  " << t_tot3 << std::endl;

  }

  Eigen::VectorXd diff = bV2.mat() - bV.mat();

  std::cout << "Eigen::NumTraits<double>::dummy_precision() " << Eigen::NumTraits<double>::dummy_precision() << std::endl;

  double maxErr = diff.cwiseAbs().maxCoeff();
  bool ok = maxErr < 16 * Eigen::NumTraits<double>::dummy_precision();
  if (!ok) {
    std::cerr << "ERROR!! max is " << maxErr << std::endl;
    std::exit(-1);
  }
  else {
    std::cerr << "OK" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  const Problem* problemPtr = &ProblemCollections::ladybug_49_7776;
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

  std::cout << "------------------- W with blocks -------------------" << std::endl;
  testFunction< mat::BlockSparse>(problem);
  std::cout << "------------------- W sparse coeffs -------------------" << std::endl;
  testFunction< mat::BlockCoeffSparse>(problem);
  return 0;
}