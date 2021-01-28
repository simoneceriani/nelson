#pragma once
#include "SolverCholeskySparse.h"

#include "MatrixSparseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskySparse<matTypeV, T, B, NB>::init(MatType& input, const mat::VectorBlock<T, B, NB>& b) {
    _sparseWrapper.set(&input);
    _incVector.resize(b.segmentDescriptionCSPtr());

    this->_ldlt.analyzePattern(_sparseWrapper.mat());
  }

  template<int matTypeV, class T, int B, int NB>
  T SolverCholeskySparse<matTypeV, T, B, NB>::maxAbsHDiag() const {
    return _sparseWrapper.maxAbsDiag();
  }

  template<int matTypeV, class T, int B, int NB>
  bool SolverCholeskySparse<matTypeV, T, B, NB>::computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda) {
    // TODO relLambda relLambda

    _sparseWrapper.refresh();

    this->_ldlt.factorize(_sparseWrapper.mat());
    if (this->_ldlt.info() != Eigen::ComputationInfo::Success) return false;

    _incVector.mat() = this->_ldlt.solve(-b.mat());
    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }


}