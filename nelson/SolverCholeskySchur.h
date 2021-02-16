#pragma once
#include "Global.h"

#include "MatrixDenseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "DoubleSectionHessianMatrices.h"

#include <Eigen/Dense>

namespace nelson {

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    class SolverCholeskyU
  >
  class SolverCholeskySchur {

  public:

    using DoubleSectionHessianMatricesT = DoubleSectionHessianMatrices<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV>;
    using DoubleSectionHessianVectorsT = DoubleSectionHessianVectors<T, BU, BV, NBU, NBV>;

  private:
    DoubleSectionHessianVectorsT _incVector;

    SolverCholeskyU _solverU;

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