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

#include <Eigen/Dense>

namespace nelson {

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int solverVType
  >
  class SolverCholeskySchur {

  public:

    using DoubleSectionHessianMatricesT = DoubleSectionHessianMatrices<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV>;
    using DoubleSectionHessianVectorsT = DoubleSectionHessianVectors<T, BU, BV, NBU, NBV>;

  private:
    DoubleSectionHessianVectorsT _incVector;

    typename SolverTraits<solverVType>::template Solver<SingleSectionHessianTraits<matTypeV,T,BV,NBV>> _solverVMatrix;

  public:

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