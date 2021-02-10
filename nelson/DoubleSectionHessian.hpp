#pragma once
#include "DoubleSectionHessian.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

namespace nelson {
  
  template<int matTypeUv, int matTypeWv, int matTypeVv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessian<matTypeUv, matTypeWv, matTypeVv, Tv, BUv, BVv, NBUv, NBVv>::DoubleSectionHessian()
    : _chi2(0)
  {

  }

  template<int matTypeUv, int matTypeWv, int matTypeVv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessian<matTypeUv, matTypeWv, matTypeVv, Tv, BUv, BVv, NBUv, NBVv>::~DoubleSectionHessian() {

  }

  template<int matTypeUv, int matTypeWv, int matTypeVv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSectionHessian<matTypeUv, matTypeWv, matTypeVv, Tv, BUv, BVv, NBUv, NBVv>::resize(
    BlockSizeTypeParU blockSizesU, int nBlocksU,
    BlockSizeTypeParV blockSizesV, int nBlocksV,
    const mat::SparsityPattern<mat::ColMajor>::CSPtr& spU,
    const mat::SparsityPattern<mat::ColMajor>::CSPtr& spV,
    const mat::SparsityPattern<mat::ColMajor>::CSPtr& spW
  )
  {
    auto blockDescriptorU = MatTypeU::BlockDescriptor::squareMatrix(blockSizesU, nBlocksU);
    _U.resize(blockDescriptorU, spU);
    _bU.resize(blockDescriptorU.rowDescriptionCSPtr());

    auto blockDescriptorV = MatTypeV::BlockDescriptor::squareMatrix(blockSizesV, nBlocksV);
    _V.resize(blockDescriptorV, spV);
    _bV.resize(blockDescriptorV.rowDescriptionCSPtr());

    auto blockDescriptorW = typename MatTypeW::BlockDescriptor(blockDescriptorU.rowDescriptionCSPtr(), blockDescriptorV.colDescriptionCSPtr());
    _W.resize(blockDescriptorW, spW);

  }

  template<int matTypeUv, int matTypeWv, int matTypeVv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSectionHessian<matTypeUv, matTypeWv, matTypeVv, Tv, BUv, BVv, NBUv, NBVv>::clearAll() {
    _chi2 = 0;
    _U.setZero();
    _V.setZero();
    _W.setZero();
    _bU.setZero();
    _bV.setZero();
  }

  template<int matTypeUv, int matTypeWv, int matTypeVv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  Tv DoubleSectionHessian<matTypeUv, matTypeWv, matTypeVv, Tv, BUv, BVv, NBUv, NBVv>::maxAbsValBVect() const {
    return std::max(_bU.mat().cwiseAbs().maxCoeff(), _bV.mat().cwiseAbs().maxCoeff());
  }
  
}