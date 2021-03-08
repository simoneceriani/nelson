#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>
#include <fstream>

#include "nelson/MatrixDiagInv.hpp"
#include "nelson/MatrixDenseWrapper.hpp"
#include "nelson/MatrixSparseWrapper.hpp"

constexpr int BC = 3;

constexpr int matVType = mat::BlockDiagonal;

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

  typename mat::MatrixBlockIterableTypeTraits<matVType, double, mat::ColMajor, BC, BC, mat::Dynamic, mat::Dynamic>::MatrixType V;
  mat::SparsityPattern<mat::ColMajor>::SPtr spV(new mat::SparsityPattern<mat::ColMajor>(problem.nPoints, problem.nPoints));
  spV->setDiagonal();
  mat::MatrixBlockDescriptor<BC, BC, mat::Dynamic, mat::Dynamic> blockDescriptorV(BC, problem.nPoints, BC, problem.nPoints);
  V.resize(blockDescriptorV, spV);
  V.setZero();
  for (int bi = 0; bi < V.nonZeroBlocks(); bi++) {
    V.blockByUID(bi).setRandom();
  }

  mat::VectorBlock<double, BC, mat::Dynamic> b, bv;
  b.resize(blockDescriptorV.rowDescriptionCSPtr());
  b.mat().setRandom();
  bv.resize(blockDescriptorV.rowDescriptionCSPtr());

  nelson::MatrixDiagInv<double, BC, mat::Dynamic, mat::BlockDiagonal> mwvinv;
  const int half_maxthreads = nelson::ParallelExecSettings::maxSupportedThreads() / 2;
  const int maxthreads = nelson::ParallelExecSettings::maxSupportedThreads();



  {
    auto t0 = std::chrono::steady_clock::now();
    mwvinv.init(V);
    auto t1 = std::chrono::steady_clock::now();
    mwvinv.settings().blockInversion.setSingleThread();
    mwvinv.compute(V, 0, 0);
    auto t2 = std::chrono::steady_clock::now();
    mwvinv.settings().blockInversion.setNumThreads(half_maxthreads);
    mwvinv.compute(V, 0, 0);
    auto t3 = std::chrono::steady_clock::now();
    mwvinv.settings().blockInversion.setNumThreadsMax();
    mwvinv.compute(V, 0, 0);
    auto t4 = std::chrono::steady_clock::now();

    double t_prep = std::chrono::duration<double>(t1 - t0).count();
    double t_mult = std::chrono::duration<double>(t2 - t1).count();
    double t_tot = std::chrono::duration<double>(t2 - t0).count();
    double t_mult2 = std::chrono::duration<double>(t3 - t2).count();
    double t_mult3 = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "*** inversion time **" << std::endl;
    std::cout << "t_prep " << t_prep << std::endl;
    std::cout << "t_mult " << t_mult << std::endl;
    std::cout << "t_tot  " << t_tot << std::endl;
    std::cout << "t_mult(" << half_maxthreads << ") " << t_mult2 << std::endl;
    std::cout << "t_mult(" << maxthreads << ") " << t_mult3 << std::endl;

  }

  {
    auto t1 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setSingleThread();
    mwvinv.rightMultVector(b, bv);
    auto t2 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setNumThreads(half_maxthreads);
    mwvinv.rightMultVector(b, bv);
    auto t3 = std::chrono::steady_clock::now();
    mwvinv.settings().rightVectorMult.setNumThreadsMax();
    mwvinv.rightMultVector(b, bv);
    auto t4 = std::chrono::steady_clock::now();

    double t_mult = std::chrono::duration<double>(t2 - t1).count();
    double t_mult2 = std::chrono::duration<double>(t3 - t2).count();
    double t_mult3 = std::chrono::duration<double>(t4 - t3).count();

    std::cout << "*** vector mult time **" << std::endl;
    std::cout << "t_mult " << t_mult << std::endl;
    std::cout << "t_mult(" << half_maxthreads << ") " << t_mult2 << std::endl;
    std::cout << "t_mult(" << maxthreads << ") " << t_mult3 << std::endl;

  }



  return 0;
}