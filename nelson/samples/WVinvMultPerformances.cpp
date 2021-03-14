#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>
#include <fstream>

#include "nelson/MatrixWVinvMultiplier.hpp"
#include "nelson/MatrixDenseWrapper.hpp"
#include "nelson/MatrixSparseWrapper.hpp"

constexpr int BR = 6;
constexpr int BC = 3;

template<int matVType, int matWType>
void testFunction(const Problem& problem) {
  typename mat::MatrixBlockIterableTypeTraits<matVType, double, mat::ColMajor, BC, BC, mat::Dynamic, mat::Dynamic>::MatrixType V;
  mat::SparsityPattern<mat::ColMajor>::SPtr spV(new mat::SparsityPattern<mat::ColMajor>(problem.nPoints, problem.nPoints));
  spV->setDiagonal();

  V.resize(mat::MatrixBlockDescriptor<BC, BC, mat::Dynamic, mat::Dynamic>(BC, problem.nPoints, BC, problem.nPoints), spV);
  V.setZero();
  for (int bi = 0; bi < V.nonZeroBlocks(); bi++) {
    V.blockByUID(bi).setRandom();
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

  mat::VectorBlock<double, BC, mat::Dynamic> b;
  b.resize(V.blockDescriptor().rowDescriptionCSPtr());
  b.mat().setRandom();

  mat::VectorBlock<double, BR, mat::Dynamic> bv;
  bv.resize(W.blockDescriptor().rowDescriptionCSPtr());
  bv.mat().setRandom();


  nelson::MatrixWVinvMultiplier<matWType, matVType, double, BR, BC, mat::Dynamic, mat::Dynamic> mwvinv;
  const int half_maxthreads = nelson::ParallelExecSettings::maxSupportedThreads() / 2;
  const int maxthreads = nelson::ParallelExecSettings::maxSupportedThreads();



  // with multiplier
  {
    auto t0 = std::chrono::steady_clock::now();
    mwvinv.prepare(W);
    auto t1 = std::chrono::steady_clock::now();
    mwvinv.settings().multiplication.setSingleThread();
    mwvinv.multiply(W, V);
    auto t2 = std::chrono::steady_clock::now();
    mwvinv.settings().multiplication.setNumThreads(half_maxthreads);
    mwvinv.multiply(W, V);
    auto t3 = std::chrono::steady_clock::now();
    mwvinv.settings().multiplication.setNumThreadsMax();
    mwvinv.multiply(W, V);
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
  nelson::SparseWrapper<matVType, double, mat::ColMajor, BC, BC, mat::Dynamic, mat::Dynamic> V_Wrap;
  V_Wrap.set(&V);
  nelson::SparseWrapper<matWType, double, mat::RowMajor, BR, BC, mat::Dynamic, mat::Dynamic> W_Wrap;
  W_Wrap.set(&W);
  Eigen::SparseMatrix<double, mat::RowMajor> WVinv;
  {
    auto t0 = std::chrono::steady_clock::now();
    WVinv = W_Wrap.mat() * V_Wrap.mat();
    auto t1 = std::chrono::steady_clock::now();
    double t_tot = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "t_tot  " << t_tot << std::endl;

    auto t20 = std::chrono::steady_clock::now();
    WVinv = W_Wrap.mat() * V_Wrap.mat();
    auto t21 = std::chrono::steady_clock::now();
    double t_tot2 = std::chrono::duration<double>(t21 - t20).count();
    std::cout << "t_tot(2)  " << t_tot2 << std::endl;
    WVinv = W_Wrap.mat() * V_Wrap.mat();
    auto t22 = std::chrono::steady_clock::now();
    double t_tot3 = std::chrono::duration<double>(t22 - t21).count();
    std::cout << "t_tot(3)  " << t_tot3 << std::endl;

  }

  // checks 
  nelson::SparseWrapper<matWType, double, mat::RowMajor, BR, BC, mat::Dynamic, mat::Dynamic> WVinv_wrap;
  WVinv_wrap.set(&mwvinv.result());

  Eigen::SparseMatrix<double, mat::RowMajor> diff = (WVinv_wrap.mat() - WVinv).template triangularView<Eigen::Upper>();

  std::cout << "Eigen::NumTraits<double>::dummy_precision() " << Eigen::NumTraits<double>::dummy_precision() << std::endl;

  double maxErr = diff.coeffs().cwiseAbs().maxCoeff();
  bool ok = (diff.coeffs().cwiseAbs() < 16 * Eigen::NumTraits<double>::dummy_precision()).all();
  if (false) {
    std::ofstream f;
    f.open("loadWWt.m");
    f << "WWt = [" << std::endl;
    f << WVinv_wrap.mat() << std::endl;
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

  // with multiplier, vector
  {
    auto t1 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setSingleThread();
    mwvinv.rightMultVectorSub(b, bv);
    auto t2 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setNumThreads(half_maxthreads);
    mwvinv.rightMultVectorSub(b, bv);
    auto t3 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setNumThreadsMax();
    mwvinv.rightMultVectorSub(b, bv);
    auto t4 = std::chrono::steady_clock::now();

    double t_mult = std::chrono::duration<double>(t2 - t1).count();
    double t_mult2 = std::chrono::duration<double>(t3 - t2).count();
    double t_mult3 = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "**** VECTOR ****" << std::endl;
    std::cout << "t_mult " << t_mult << std::endl;
    std::cout << "t_mult(" << half_maxthreads << ") " << t_mult2 << std::endl;
    std::cout << "t_mult(" << maxthreads << ") " << t_mult3 << std::endl;
  }
}

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

  std::cout << "--------- V BlockDiagonal, W BlockCoeffSparse" << std::endl;
  testFunction<mat::BlockDiagonal, mat::BlockCoeffSparse>(problem);
  std::cout << std::endl;
  
  std::cout << "--------- V SparseCoeffBlockDiagonal, W BlockCoeffSparse" << std::endl;
  testFunction<mat::SparseCoeffBlockDiagonal, mat::BlockCoeffSparse>(problem);
  std::cout << std::endl;

  std::cout << "--------- V BlockDiagonal, W BlockSparse" << std::endl;
  testFunction<mat::BlockDiagonal, mat::BlockSparse>(problem);
  std::cout << std::endl;

  std::cout << "--------- V SparseCoeffBlockDiagonal, W BlockSparse" << std::endl;
  testFunction<mat::SparseCoeffBlockDiagonal, mat::BlockSparse>(problem);
  std::cout << std::endl;


  return 0;
}