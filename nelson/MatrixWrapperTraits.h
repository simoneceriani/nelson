#pragma once
#include "Global.h"

#include "mat/Global.h"

#include "MatrixSparseWrapper.h"
#include "MatrixDenseWrapper.h"

namespace nelson {

  constexpr int matrixWrapperDense  = 1;
  constexpr int matrixWrapperSparse = 2;

  template<int wrapperType>
  struct MatrixWrapperTraits
  {

  };

  template<>
  struct MatrixWrapperTraits<matrixWrapperDense>
  {
    template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
    using Wrapper = DenseWrapper<matType, T, Ordering, BR, BC, NBR, NBC>;
  };

  template<>
  struct MatrixWrapperTraits<matrixWrapperSparse>
  {
    template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
    using Wrapper = SparseWrapper<matType, T, Ordering, BR, BC, NBR, NBC>;
  };


}