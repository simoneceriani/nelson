#pragma once
#include "Global.h"

#include "MatrixDenseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include <Eigen/Dense>

namespace nelson {
  
  template<class EigenMatType>
  class SolverCholeskyEigenDense {
    Eigen::LDLT<EigenMatType, Eigen::Upper> _ldlt;
  public:
    void init(EigenMatType& mat);

    bool factorize(EigenMatType& matInput);

    template<class Derived1, class Derived2>
    void solve(const Eigen::MatrixBase<Derived1>& b, Eigen::MatrixBase<Derived2>& x);
  };
  
  //--------------------------------------------------------------------------------------

  template<int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SolverCholeskyDense {

  public:

    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using DenseWrapperT = DenseSquareWrapper<matTypeV, T, mat::ColMajor, B, NB>;
    using DiagType = typename DenseWrapperT::DiagType;

    using Type = T;

    static constexpr bool hasSettings = false;

  private:

    DenseWrapperT _denseWrapper;
    DiagType _diagBackup;

    VecType _incVector;
    Eigen::LDLT<typename DenseWrapperT::MatOutputType, Eigen::Upper> _ldlt;

  public:

    void init(MatType& input, const mat::VectorBlock<T, B, NB>& b);

    T maxAbsHDiag() const;

    bool computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda);

    const VecType& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.mat().squaredNorm();
    }

    template<class Derived>
    const Eigen::Solve<Eigen::LDLT<typename SolverCholeskyDense<matTypeV, T, B, NB>::DenseWrapperT::MatOutputType, Eigen::Upper>, Derived> solve(const Eigen::MatrixBase<Derived>& b) const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> solve(const Eigen::SparseMatrix<T>& b) const;

  };

}