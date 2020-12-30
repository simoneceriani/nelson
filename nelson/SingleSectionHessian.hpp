#pragma once
#include "SingleSectionHessian.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matType, class T, int B, int NB>
  SingleSectionHessian<matType, T, B, NB>::SingleSectionHessian() 
    : _chi2(0)
  {

  }

  template<int matType, class T, int B, int NB>
  SingleSectionHessian<matType, T, B, NB>::~SingleSectionHessian() {

  }

  template<int matType, class T, int B, int NB>
  void SingleSectionHessian<matType, T, B, NB>::resize(BlockSizeTypePar blockSizes, int nBlocks, const mat::SparsityPattern<mat::ColMajor>::CSPtr& sp) {
    auto blockDescriptor = MatType::BlockDescriptor::squareMatrix(blockSizes, nBlocks);
    _H.resize(blockDescriptor, sp);
    _b.resize(blockDescriptor.rowDescriptionCSPtr());
  }

  template<int matType, class T, int B, int NB>
  void SingleSectionHessian<matType, T, B, NB>::clearAll() {
    clearChi2();
    _H.setZero();
    _b.setZero();
  }


}