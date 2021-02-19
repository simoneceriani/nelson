#pragma once
#include "SolverCholeskyDense.h"

#include "MatrixDenseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {
  
  template<class EigenMatType>
  void SolverCholeskyEigenDense<EigenMatType>::init(EigenMatType& mat) {

  }

  template<class EigenMatType>
  bool SolverCholeskyEigenDense<EigenMatType>::factorize(EigenMatType& matInput) {
    this->_ldlt.compute(matInput);
    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<class EigenMatType>
  template<class Derived1, class Derived2>
  void SolverCholeskyEigenDense<EigenMatType>::solve(const Eigen::MatrixBase<Derived1>& b, Eigen::MatrixBase<Derived2>& x) {
    x = this->_ldlt.solve(b);
  }
  
  //---------------------------------------------------------------------------------------------------

  template<int matTypeV, class T, int B, int NB>
  void SolverCholeskyDense<matTypeV, T, B, NB>::init(MatType& input, const mat::VectorBlock<T, B, NB>& b) {
    _denseWrapper.set(&input);
    _incVector.resize(b.segmentDescriptionCSPtr());
  }

  template<int matTypeV, class T, int B, int NB>
  T SolverCholeskyDense<matTypeV, T, B, NB>::maxAbsHDiag() const {
    return _denseWrapper.maxAbsDiag();
  }

  template<int matTypeV, class T, int B, int NB>
  bool SolverCholeskyDense<matTypeV, T, B, NB>::computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda) {
    _denseWrapper.refresh();

    if (relLambda != 0 || absLambda != 0) {
      if (_diagBackup.size() != _denseWrapper.mat().rows()) {
        _diagBackup.resize(_denseWrapper.mat().rows());
      }
      // backup diagonal
      _denseWrapper.diagonalCopy(_diagBackup);
      // change the diagonal values
      _denseWrapper.setDiagonal(
        (_diagBackup.array() * (1 + relLambda)).array() + absLambda
      );
    }


    this->_ldlt.compute(_denseWrapper.mat());

    if (this->_ldlt.info() == Eigen::ComputationInfo::Success) {
      _incVector.mat() = this->_ldlt.solve(-b.mat());
    }

    // restore diagonal values
    if (relLambda != 0 || absLambda != 0) {
      _denseWrapper.setDiagonal(_diagBackup);
    }

    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<int matTypeV, class T, int B, int NB>
  template<class Derived>
  const Eigen::Solve<Eigen::LDLT<typename SolverCholeskyDense<matTypeV, T, B, NB>::DenseWrapperT::MatOutputType, Eigen::Upper>, Derived> SolverCholeskyDense<matTypeV, T, B, NB>::solve(const Eigen::MatrixBase<Derived>& b) const {
    return this->_ldlt.solve(b);
  }

  template<int matTypeV, class T, int B, int NB>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> SolverCholeskyDense<matTypeV, T, B, NB>::solve(const Eigen::SparseMatrix<T>& b) const {
    return this->_ldlt.solve(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(b));
  }

}
