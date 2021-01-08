#pragma once
#include "Global.h"

#include "MatrixDenseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include <Eigen/Dense>

namespace nelson {

  template<int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SolverCholeskyDense {

  public:

    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using DenseWrapperT = DenseWrapper<matTypeV, T, mat::ColMajor, B, B, NB, NB>;

  private:

    DenseWrapperT _denseWrapper;
    VecType _incVector;
    Eigen::LDLT<typename DenseWrapperT::MatOutputType, Eigen::Upper> _ldlt;

  public:

    void init(MatType& input, const mat::VectorBlock<T, B, NB>& b);

    bool computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda);

    const VecType& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.mat().squaredNorm();
    }

  };

}