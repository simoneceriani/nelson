#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

namespace nelson {

  template<int matTypeV, class Tv, int Bv, int NBv> 
  struct SingleSectionHessianTraits {
      static constexpr int matType = matTypeV;
      using Type = Tv;
      static constexpr int B = Bv;
      static constexpr int NB = NBv;

  };


  template<int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SingleSectionHessian {
  public:
    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;

    using Traits = SingleSectionHessianTraits<matTypeV, T, B, NB>;

    
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using BlockSizeTypePar = typename MatType::RowTraits::BlockSizeTypePar;

  private:
    MatType _H;
    VecType _b;
    double _chi2;

  public:
    SingleSectionHessian();
    virtual ~SingleSectionHessian();

    void resize(BlockSizeTypePar blockSizes, int nBlocks, const mat::SparsityPattern<mat::ColMajor>::CSPtr& sp);

    inline void setChi2(double chi2) {
      this->_chi2 = chi2;
    }

    inline double chi2() const {
      return _chi2;
    }

    inline const MatType& H() const {
      return _H;
    }

    inline MatType& H() {
      return _H;
    }

    inline const VecType& b() const {
      return _b;
    }

    inline VecType& b() {
      return _b;
    }

    void clearAll();

    T maxAbsValBVect() const;

  };
}