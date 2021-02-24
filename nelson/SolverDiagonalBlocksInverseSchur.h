#pragma once
#include "Global.h"

#include "MatrixDenseWrapper.h"
#include "MatrixSparseWrapper.h"

#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "DoubleSectionHessianMatrices.h"

#include "SingleSectionHessian.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"

#include "SolverTraitsBase.h"
#include "MatrixWrapperTraits.h"

#include <Eigen/Dense>

#include "ParallelExecHelper.h"

#include <chrono>

namespace nelson {

  struct SolverDiagonalBlocksInverseSchurIterationTimeStat {
    std::chrono::steady_clock::time_point t0_startIteration;
    std::chrono::steady_clock::time_point t1_VInvComputed;
    std::chrono::steady_clock::time_point t2_WRefreshed;
    std::chrono::steady_clock::time_point t3_URefreshed;
    std::chrono::steady_clock::time_point t4_bSComputed;
    std::chrono::steady_clock::time_point t5_SComputed;
    std::chrono::steady_clock::time_point t6_SSolveInit;
    std::chrono::steady_clock::time_point t7_SFactorized;
    std::chrono::steady_clock::time_point t8_bUComputed;
    std::chrono::steady_clock::time_point t9_bVtildeComputed;
    std::chrono::steady_clock::time_point t10_bVComputed;
    std::string toString(const std::string & linePrefix = "") const;
  };

  struct SolverDiagonalBlocksInverseSchurTimeStats {
    std::chrono::steady_clock::time_point startInit;
    std::chrono::steady_clock::time_point endInit;
    std::vector< SolverDiagonalBlocksInverseSchurIterationTimeStat> iterations;
    SolverDiagonalBlocksInverseSchurIterationTimeStat& lastIteration() {
      return iterations.back();
    }
    void addIteration() { iterations.push_back(SolverDiagonalBlocksInverseSchurIterationTimeStat()); }
    std::string toString(const std::string& linePrefix = "") const;
  };

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType, 
    int choleskyOrderingS
  >
  class SolverDiagonalBlocksInverseSchur {

  public:

    using DoubleSectionHessianMatricesT = DoubleSectionHessianMatrices<matTypeU, mat::BlockDiagonal, matTypeW, T, BU, BV, NBU, NBV>;
    using DoubleSectionHessianVectorsT = DoubleSectionHessianVectors<T, BU, BV, NBU, NBV>;

    using Type = T;

    static constexpr bool hasSettings = true;
    using Settings = ParallelExecSettings;

  private:
    DoubleSectionHessianVectorsT _incVector;

    typename DoubleSectionHessianVectorsT::VecTypeU::StorageType _bS;
    typename DoubleSectionHessianVectorsT::VecTypeV::StorageType _bVtilde;

    typename mat::SparseCoeffMatrixBlock<T, mat::ColMajor, BV, BV, NBV, NBV> _matrixVInv;
    typename MatrixWrapperTraits<wrapperWType>::template Wrapper<matTypeW, T, mat::ColMajor, BU, BV, NBU, NBV> _matrixW;

    using MatrixUType = typename MatrixWrapperTraits<wrapperUType>::template Wrapper<matTypeU, T, mat::ColMajor, BU, BU, NBU, NBU>;
    MatrixUType _matrixU;
    typename MatrixUType::MatOutputType _matrixS;

    typename SolverTraits<wrapperUType>::template SolverEigen<typename MatrixUType::MatOutputType, choleskyOrderingS> _solverS;

    bool _firstTime;
    T _v_maxAbsHDiag;

    Settings _settings;
    SolverDiagonalBlocksInverseSchurTimeStats _timeStats;

    void computeVInv(DoubleSectionHessianMatricesT& input, T relLambda, T absLambda);

  public:
    SolverDiagonalBlocksInverseSchur();

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    const SolverDiagonalBlocksInverseSchurTimeStats& timeStats() const {
      return _timeStats;
    }

    void init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b);

    T maxAbsHDiag() const;

    bool computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda);

    const DoubleSectionHessianVectorsT& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.bU().mat().squaredNorm() + _incVector.bV().mat().squaredNorm();
    }
  };


}