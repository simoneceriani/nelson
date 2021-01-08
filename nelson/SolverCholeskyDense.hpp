#pragma once
#include "SolverCholeskyDense.h"

#include "MatrixDenseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskyDense<matTypeV, T, B, NB>::init(MatType& input, const mat::VectorBlock<T, B, NB>& b) {
    _denseWrapper.set(&input);
    _incVector.resize(b.segmentDescriptionCSPtr());
  }

  template<int matTypeV, class T, int B, int NB>
  bool SolverCholeskyDense<matTypeV, T, B, NB>::computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda) {
    // TODO relLambda relLambda

    _denseWrapper.refresh();

    this->_ldlt.compute(_denseWrapper.mat());
    _incVector.mat() = this->_ldlt.solve(-b.mat());

    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }


}