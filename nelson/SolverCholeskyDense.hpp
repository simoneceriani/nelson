#pragma once
#include "SolverCholeskyDense.h"

#include "MatrixDenseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskyDense<matTypeV, T, B, NB>::init(MatType& input, const mat::VectorBlock<T, B, NB>& b) {
    _denseWrapper.set(&input);
    _diagBackup.resize(_denseWrapper.mat().rows());
    _incVector.resize(b.segmentDescriptionCSPtr());
  }

  template<int matTypeV, class T, int B, int NB>
  T SolverCholeskyDense<matTypeV, T, B, NB>::maxAbsHDiag() const {
    return _denseWrapper.maxAbsDiag();
  }

  template<int matTypeV, class T, int B, int NB>
  bool SolverCholeskyDense<matTypeV, T, B, NB>::computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda) {
    _denseWrapper.refresh();

    if (relLambda != 0 || absLambda != 0) {
      // backup diagonal
      _denseWrapper.diagonalCopy(_diagBackup);
      // change the diagonal values
      _denseWrapper.setDiagonal(((_diagBackup.array() * 1 + relLambda).array() + absLambda));
    }


    this->_ldlt.compute(_denseWrapper.mat());
    _incVector.mat() = this->_ldlt.solve(-b.mat());

    // restore diagonal values
    if (relLambda != 0 || absLambda != 0) {
      _denseWrapper.setDiagonal(_diagBackup);
    }

    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<int matTypeV, class T, int B, int NB>
  template<class Derived>
  void SolverCholeskyDense<matTypeV, T, B, NB>::solve(const Eigen::MatrixBase<Derived>& B) const {
    this->_ldlt.solve(B);
  }


}