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
  class SolverCholeskyDenseBase {

  public:

    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using DenseWrapperT = DenseSquareWrapper<matTypeV, T, mat::ColMajor, B, NB>;
    using DiagType = typename DenseWrapperT::DiagType;

    using Type = T;

    static constexpr bool hasSettings = false;
    class Settings {};

  private:

    DenseWrapperT _denseWrapper;
    DiagType _diagBackup;

    Eigen::LDLT<typename DenseWrapperT::MatOutputType, Eigen::Upper> _ldlt;

  public:

    virtual void init(MatType& input, const VecType& b);

    T maxAbsHDiag() const;

    bool computeIncrement(MatType& input, const VecType& b, T relLambda, T absLambda, VecType & result);

    template<class Derived>
    const Eigen::Solve<Eigen::LDLT<typename SolverCholeskyDenseBase<matTypeV, T, B, NB>::DenseWrapperT::MatOutputType, Eigen::Upper>, Derived> solve(const Eigen::MatrixBase<Derived>& b) const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> solve(const Eigen::SparseMatrix<T>& b) const;

  };

  //--------------------------------------------------------------------------------------
  template<int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SolverCholeskyDense : public SolverCholeskyDenseBase<matTypeV, T, B, NB> {
  public:
    using VecType = typename SolverCholeskyDenseBase<matTypeV, T, B, NB>::VecType;
    using MatType = typename SolverCholeskyDenseBase<matTypeV, T, B, NB>::MatType;
  private:

    VecType _incVector;

  public:

    void init(MatType& input, const VecType & b) override;

    bool computeIncrement(MatType& input, const VecType & b, T relLambda, T absLambda);

    const VecType& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.mat().squaredNorm();
    }

  };

}