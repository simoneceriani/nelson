#pragma once
#include "SingleSectionHessian.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

namespace nelson {

  template<int matType, class T, int B, int NB>
  SingleSectionHessianEval<matType, T, B, NB>::SingleSectionHessianEval() 
    : _chi2(0)
  {

  }

  template<int matType, class T, int B, int NB>
  SingleSectionHessianEval<matType, T, B, NB>::~SingleSectionHessianEval() {

  }

  template<int matType, class T, int B, int NB>
  void SingleSectionHessianEval<matType, T, B, NB>::resize(BlockSizeTypePar blockSizes, int nBlocks, const mat::SparsityPattern<mat::ColMajor>& sp) {
    auto blockDescriptor = MatType::BlockDescriptor::squareMatrix(blockSizes, nBlocks);
    _H.resize(blockDescriptor, sp);
    _b.resize(blockDescriptor.rowDescriptionCSPtr());
  }

  template<int matType, class T, int B, int NB>
  void SingleSectionHessianEval<matType, T, B, NB>::clearAll() {
    clearChi2();
    _H.setZero();
    _b.setZero();
  }


}