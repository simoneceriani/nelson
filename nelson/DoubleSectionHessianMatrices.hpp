#pragma once
#include "DoubleSectionHessianMatrices.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include "mat/SparseCoeffDiagonalMatrixBlock.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {
  
  template<int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessianMatrices<matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::DoubleSectionHessianMatrices() {
    
  }
  
  template<int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessianMatrices<matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::~DoubleSectionHessianMatrices() {
    
  }

//----------------------------------------------------------------------
  template<class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessianVectors<Tv, BUv, BVv, NBUv, NBVv>::DoubleSectionHessianVectors() {
    
  }
  
  template<class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSectionHessianVectors<Tv, BUv, BVv, NBUv, NBVv>::~DoubleSectionHessianVectors() {
    
  }

  
}