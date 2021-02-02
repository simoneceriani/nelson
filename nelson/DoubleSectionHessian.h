#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

namespace nelson {

  /*
  U  W
  W' V
  */

  template<int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  struct DoubleSectionHessianTraits {
    static constexpr int matTypeU = matTypeUv;
    static constexpr int matTypeW = matTypeWv;
    static constexpr int matTypeV = matTypeVv;

    using Type = Tv;
    
    static constexpr int BU = BUv;
    static constexpr int BV = BVv;
    static constexpr int NBU = NBUv;
    static constexpr int NBV = NBVv;
  };


  template<int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv = mat::Dynamic, int NBVv = mat::Dynamic>
  class DoubleSectionHessian {
  public:
    using MatTraitsU = mat::MatrixBlockIterableTypeTraits<matTypeUv, Tv, mat::ColMajor, BUv, BUv, NBUv, NBUv>;
    using MatTraitsV = mat::MatrixBlockIterableTypeTraits<matTypeVv, Tv, mat::ColMajor, BVv, BVv, NBVv, NBVv>;
    using MatTraitsW = mat::MatrixBlockIterableTypeTraits<matTypeWv, Tv, mat::ColMajor, BUv, BVv, NBUv, NBVv>;

    using Traits = DoubleSectionHessianTraits<matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>;

    using MatTypeU = typename MatTraitsU::MatrixType;
    using MatTypeV = typename MatTraitsV::MatrixType;
    using MatTypeW = typename MatTraitsW::MatrixType;

    using VecTypeU = mat::VectorBlock<Tv, BUv, NBUv>;
    using VecTypeV = mat::VectorBlock<Tv, BVv, NBVv>;

    using BlockSizeTypeParU = typename MatTypeU::RowTraits::BlockSizeTypePar;
    using BlockSizeTypeParV = typename MatTypeV::RowTraits::BlockSizeTypePar;

  private:
    MatTypeU _U;
    MatTypeV _V;
    MatTypeW _W;

    VecTypeU _bU;
    VecTypeV _bV;

    double _chi2;

  public:
    DoubleSectionHessian();
    virtual ~DoubleSectionHessian();

    void resize(
      BlockSizeTypeParU blockSizesU, int nBlocksU,
      BlockSizeTypeParV blockSizesV, int nBlocksV,
      const mat::SparsityPattern<mat::ColMajor>::CSPtr& spU,
      const mat::SparsityPattern<mat::ColMajor>::CSPtr& spV,
      const mat::SparsityPattern<mat::ColMajor>::CSPtr& spW
    );

    inline void setChi2(double chi2) {
      this->_chi2 = chi2;
    }

    inline double chi2() const {
      return _chi2;
    }

    inline const MatTypeU& U() const {
      return _U;
    }
    inline const MatTypeV& V() const {
      return _V;
    }
    inline const MatTypeU& W() const {
      return _W;
    }

    inline MatTypeU& U() {
      return _U;
    }
    inline MatTypeV& V() {
      return _V;
    }
    inline MatTypeU& W() {
      return _W;
    }

    inline const VecTypeU& bU() const {
      return _bU;
    }
    inline const VecTypeV& bV() const {
      return _bV;
    }

    inline VecTypeU& bU() {
      return _bU;
    }
    inline VecTypeV& bV() {
      return _bV;
    }


    void clearAll();

    Tv maxAbsValBVect() const;

  };
  
}