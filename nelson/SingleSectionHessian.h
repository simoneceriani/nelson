#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

namespace nelson {

  template<int matType, class T, int B, int NB = mat::Dynamic>
  class SingleSectionHessian {
  public:
    using MatTraits = mat::MatrixBlockIterableTypeTraits<matType, T, mat::ColMajor, B, B, NB, NB>;

    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using BlockSizeTypePar = typename MatType::RowTraits::BlockSizeTypePar;

  private:
    MatType _H;
    VecType _b;
    T _chi2;

  public:
    SingleSectionHessian();
    virtual ~SingleSectionHessian();

    void resize(BlockSizeTypePar blockSizes, int nBlocks, const mat::SparsityPattern<mat::ColMajor>& sp);

    inline void clearChi2() {
      this->_chi2 = 0;
    }

    inline T chi2() const {
      return _chi2;
    }

    inline const MatType& H() const {
      return _H;
    }

    inline MatType& H() {
      return _H;
    }

    void clearAll();

  };
}