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

namespace nelson {

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

    void computeVInv(DoubleSectionHessianMatricesT& input, T relLambda, T absLambda);

  public:
    SolverDiagonalBlocksInverseSchur();

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

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