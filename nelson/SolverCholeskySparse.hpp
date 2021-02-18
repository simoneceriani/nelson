#pragma once
#include "SolverCholeskySparse.h"

#include "MatrixSparseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskySparse<matTypeV, T, B, NB>::init(MatType& input, const mat::VectorBlock<T, B, NB>& b) {
    _sparseWrapper.set(&input);
    _diagBackup.resize(_sparseWrapper.mat().rows());
    _incVector.resize(b.segmentDescriptionCSPtr());

    this->_ldlt.analyzePattern(_sparseWrapper.mat());
  }

  template<int matTypeV, class T, int B, int NB>
  T SolverCholeskySparse<matTypeV, T, B, NB>::maxAbsHDiag() const {
    return _sparseWrapper.maxAbsDiag();
  }

  template<int matTypeV, class T, int B, int NB>
  bool SolverCholeskySparse<matTypeV, T, B, NB>::computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda) {
    _sparseWrapper.refresh();

    if (relLambda != 0 || absLambda != 0) {
      // backup diagonal
      _sparseWrapper.diagonalCopy(_diagBackup);
      // change the diagonal values
      _sparseWrapper.setDiagonal(((_diagBackup.array() * 1 + relLambda).array() + absLambda));
    }

    this->_ldlt.factorize(_sparseWrapper.mat());
    if (this->_ldlt.info() != Eigen::ComputationInfo::Success) return false;

    _incVector.mat() = this->_ldlt.solve(-b.mat());

    // restore diagonal values
    if (relLambda != 0 || absLambda != 0) {
      _sparseWrapper.setDiagonal(_diagBackup);
    }

    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskySparse<matTypeV, T, B, NB>::solve(const Eigen::SparseMatrix<T>& b) const {
    this->_ldlt.solve(b);
  }

}