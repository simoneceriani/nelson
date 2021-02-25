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

#include <chrono>

namespace nelson {

  struct SolverCholeskySchurIterationTimeStat {
    std::chrono::steady_clock::time_point t0_startIteration;
    std::chrono::steady_clock::time_point t1_VInvbVComputed;
    std::chrono::steady_clock::time_point t2_WRefreshed;
    std::chrono::steady_clock::time_point t3_URefreshed;
    std::chrono::steady_clock::time_point t4_bSComputed;
    std::chrono::steady_clock::time_point t5_SComputed;
    std::chrono::steady_clock::time_point t6_SSolveInit;
    std::chrono::steady_clock::time_point t7_SFactorized;
    std::chrono::steady_clock::time_point t8_bUComputed;
    std::chrono::steady_clock::time_point t9_bVtildeComputed;
    std::chrono::steady_clock::time_point t10_bVComputed;
    std::string toString(const std::string& linePrefix = "") const;
  };

  struct SolverCholeskySchurTimeStat {
    std::chrono::steady_clock::time_point startInit;
    std::chrono::steady_clock::time_point t_initVSolver;
    std::chrono::steady_clock::time_point endInit;
    std::vector< SolverCholeskySchurIterationTimeStat> iterations;
    SolverCholeskySchurIterationTimeStat& lastIteration() {
      return iterations.back();
    }
    void addIteration() { iterations.push_back(SolverCholeskySchurIterationTimeStat()); }
    std::string toString(const std::string& linePrefix = "") const;
  };


  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType,
    int wrapperWType,
    int solverVType, 
    int choleskyOrderingS,
    int choleskyOrderingV
  >
  class SolverCholeskySchur {

  public:

    using DoubleSectionHessianMatricesT = DoubleSectionHessianMatrices<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV>;
    using DoubleSectionHessianVectorsT = DoubleSectionHessianVectors<T, BU, BV, NBU, NBV>;

    using Type = T;

    static constexpr bool hasSettings = false;

  private:
    DoubleSectionHessianVectorsT _incVector;

    typename DoubleSectionHessianVectorsT::VecTypeU::StorageType _bS;
    typename DoubleSectionHessianVectorsT::VecTypeV::StorageType _bVtilde;

    typename SolverTraits<solverVType>::template Solver<SingleSectionHessianTraits<matTypeV,T,BV,NBV>, choleskyOrderingV> _solverVMatrix;
    typename MatrixWrapperTraits<wrapperWType>::template Wrapper<matTypeW, T, mat::RowMajor, BU, BV, NBU, NBV> _matrixW;

    using MatrixUType = typename MatrixWrapperTraits<wrapperUType>::template Wrapper<matTypeU, T, mat::ColMajor, BU, BU, NBU, NBU>;
    MatrixUType _matrixU;
    typename MatrixUType::MatOutputType _matrixS;

    typename SolverTraits<wrapperUType>::template SolverEigen<typename MatrixUType::MatOutputType, choleskyOrderingS> _solverS;

    bool _firstTime;
    
    SolverCholeskySchurTimeStat _timeStats;

  public:
    SolverCholeskySchur();

    void init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b);

    const SolverCholeskySchurTimeStat& timeStats() const {
      return _timeStats;
    }


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