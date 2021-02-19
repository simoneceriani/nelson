#pragma once
#include "Global.h"

#include "MatrixDenseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "DoubleSectionHessianMatrices.h"

#include "SingleSectionHessian.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"

#include "SolverTraitsBase.h"
#include "MatrixWrapperTraits.h"

#include <Eigen/Dense>

namespace nelson {

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

  private:
    DoubleSectionHessianVectorsT _incVector;

    typename DoubleSectionHessianVectorsT::VecTypeU::StorageType _bS;
    typename DoubleSectionHessianVectorsT::VecTypeV::StorageType _bVtilde;

    typename SolverTraits<solverVType>::template Solver<SingleSectionHessianTraits<matTypeV,T,BV,NBV>, choleskyOrderingV> _solverVMatrix;
    typename MatrixWrapperTraits<wrapperWType>::template Wrapper<matTypeW, T, mat::ColMajor, BU, BV, NBU, NBV> _matrixW;

    using MatrixUType = typename MatrixWrapperTraits<wrapperUType>::template Wrapper<matTypeU, T, mat::ColMajor, BU, BU, NBU, NBU>;
    typename MatrixUType _matrixU;
    typename MatrixUType::MatOutputType _matrixS;

    typename SolverTraits<wrapperUType>::template SolverEigen<typename MatrixUType::MatOutputType, choleskyOrderingS> _solverS;

    bool _firstTime;

  public:
    SolverCholeskySchur();

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