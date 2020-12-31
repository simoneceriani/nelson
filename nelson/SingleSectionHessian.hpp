#pragma once
#include "SingleSectionHessian.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matTypeV, class T, int B, int NB>
  SingleSectionHessian<matTypeV, T, B, NB>::SingleSectionHessian() 
    : _chi2(0)
  {

  }

  template<int matTypeV, class T, int B, int NB>
  SingleSectionHessian<matTypeV, T, B, NB>::~SingleSectionHessian() {

  }

  template<int matTypeV, class T, int B, int NB>
  void SingleSectionHessian<matTypeV, T, B, NB>::resize(BlockSizeTypePar blockSizes, int nBlocks, const mat::SparsityPattern<mat::ColMajor>::CSPtr& sp) {
    auto blockDescriptor = MatType::BlockDescriptor::squareMatrix(blockSizes, nBlocks);
    _H.resize(blockDescriptor, sp);
    _b.resize(blockDescriptor.rowDescriptionCSPtr());
  }

  template<int matTypeV, class T, int B, int NB>
  void SingleSectionHessian<matTypeV, T, B, NB>::clearAll() {
    _chi2 = 0;
    _H.setZero();
    _b.setZero();
  }

  template<int matTypeV, class T, int B, int NB>
  T SingleSectionHessian<matTypeV, T, B, NB>::maxAbsValBVect() const {
    return _b.mat().cwiseAbs().maxCoeff();
  }

}