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

#include "MatrixDiagInv.h"
#include "MatrixWVinvMultiplier.h"
#include "MatrixWtXMultiplier.h"
#include "MatrixWWtMultiplier.h"


#include <chrono>


namespace nelson {

  struct SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat {
    std::chrono::steady_clock::time_point t0_startIteration;
    std::chrono::steady_clock::time_point t1_VInvComputed;
    std::chrono::steady_clock::time_point t2_VinvWComputed;
    std::chrono::steady_clock::time_point t3_WVinvWtComputed;
    std::chrono::steady_clock::time_point t4_bSComputed;
    std::chrono::steady_clock::time_point t5_SComputed;
    std::chrono::steady_clock::time_point t6_SSolveInit;
    std::chrono::steady_clock::time_point t7_SFactorized;
    std::chrono::steady_clock::time_point t8_bUComputed;
    std::chrono::steady_clock::time_point t9_bVtildeComputed;
    std::chrono::steady_clock::time_point t10_bVComputed;
    std::string toString(const std::string & linePrefix = "") const;
  };

  struct SolverDiagonalBlocksInverseWWtMultSchurTimeStats {
    std::chrono::steady_clock::time_point startInit;
    std::chrono::steady_clock::time_point endInit;
    std::vector< SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat> iterations;
    SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat& lastIteration() {
      return iterations.back();
    }
    void addIteration() { iterations.push_back(SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat()); }
    std::string toString(const std::string& linePrefix = "") const;
  };
  
  struct SolverDiagonalBlocksInverseWWtMultSchurIterationSettings {
    MatrixDiagInvSettings& Vinv;
    MatrixWVinvMultiplierSettings& WVinv;
    MatrixWWtMultiplierSettings& WVinvWt;
    MatrixWtXMultiplierSettings& WtX;

    SolverDiagonalBlocksInverseWWtMultSchurIterationSettings(
      MatrixDiagInvSettings& Vinv,
      MatrixWVinvMultiplierSettings& WVinv,
      MatrixWWtMultiplierSettings& WVinvWt,
      MatrixWtXMultiplierSettings& WtX
    ) : 
      Vinv(Vinv), 
      WVinv(WVinv),
      WVinvWt(WVinvWt),
      WtX(WtX)
    {
      
    }
  };
  
  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType,
    int choleskyOrderingS
  >
  class SolverDiagonalBlocksInverseWWtMultSchur {
  public:

    using DoubleSectionHessianMatricesT = DoubleSectionHessianMatrices<matTypeU, mat::BlockDiagonal, matTypeW, T, BU, BV, NBU, NBV>;
    using DoubleSectionHessianVectorsT = DoubleSectionHessianVectors<T, BU, BV, NBU, NBV>;

    using Type = T;

    static constexpr bool hasSettings = true;
    using Settings = SolverDiagonalBlocksInverseWWtMultSchurIterationSettings;    

  private:
    DoubleSectionHessianVectorsT _incVector;

    typename DoubleSectionHessianVectorsT::VecTypeU _bS;
    typename DoubleSectionHessianVectorsT::VecTypeV _bVtilde;

    static constexpr int matSType = MatrixWrapperTraits<SType>::template Wrapper<matTypeU, T, mat::ColMajor, BU, BU, NBU, NBU>::matOutputType;

    MatrixDiagInv<double, BV, NBV, mat::BlockDiagonal> _Vinv;
    MatrixWVinvMultiplier<matTypeW, T, BU, BV, NBU, NBV> _WVinv;
    MatrixWWtMultiplier<matSType, T, BU, NBU> _wwtMult;
    MatrixWtXMultiplier<matTypeW, T, BU, BV, NBU, NBV> _WtX;
    
    
    typename SolverTraits<SType>::template Solver<SingleSectionHessianTraits<matSType, T, BU, NBU>, choleskyOrderingS> _solverS;

    T _uv_maxAbsHDiag;

    Settings _settings;
    SolverDiagonalBlocksInverseWWtMultSchurTimeStats _timeStats;
   
  public:
    SolverDiagonalBlocksInverseWWtMultSchur();

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    const SolverDiagonalBlocksInverseWWtMultSchurTimeStats& timeStats() const {
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