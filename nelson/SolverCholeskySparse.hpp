#pragma once
#include "SolverCholeskySparse.h"

#include "MatrixSparseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {

  template<class EigenMatType, class EigenOrderingMethod>
  void SolverCholeskyEigenSparse<EigenMatType, EigenOrderingMethod>::init(EigenMatType& mat) {
    this->_ldlt.analyzePattern(mat);
  }

  template<class EigenMatType, class EigenOrderingMethod>
  bool SolverCholeskyEigenSparse<EigenMatType, EigenOrderingMethod>::factorize(EigenMatType& matInput) {
    this->_ldlt.factorize(matInput);
    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<class EigenMatType, class EigenOrderingMethod>
  template<class Derived1, class Derived2>
  void SolverCholeskyEigenSparse<EigenMatType, EigenOrderingMethod>::solve(const Eigen::MatrixBase<Derived1>& b, Eigen::MatrixBase<Derived2>& x) {
    x = this->_ldlt.solve(b);
  }

  //---------------------------------------------------------------------------------------------------

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  void SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::init(MatType& input, const VecType & b) {
    _sparseWrapper.set(&input);
    
    this->_ldlt.analyzePattern(_sparseWrapper.mat());
  }

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  T SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::maxAbsHDiag() const {
    return _sparseWrapper.maxAbsDiag();
  }

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  bool SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::computeIncrement(MatType& input, const VecType& b, T relLambda, T absLambda, VecType& result) {
    _sparseWrapper.refresh();

    if (relLambda != 0 || absLambda != 0) {
      if (_diagBackup.size() != _sparseWrapper.mat().rows()) {
        _diagBackup.resize(_sparseWrapper.mat().rows());
      }
      // backup diagonal
      _sparseWrapper.diagonalCopy(_diagBackup);
      // change the diagonal values
      _sparseWrapper.setDiagonal(
        (_diagBackup.array() * (1 + relLambda)).array() + absLambda
      );

    }

    this->_ldlt.factorize(_sparseWrapper.mat());

    if (this->_ldlt.info() == Eigen::ComputationInfo::Success) {
      result.mat() = this->_ldlt.solve(-b.mat());
    }

    // restore diagonal values
    if (relLambda != 0 || absLambda != 0) {
      _sparseWrapper.setDiagonal(_diagBackup);
    }

    return this->_ldlt.info() == Eigen::ComputationInfo::Success;
  }

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  const Eigen::Solve<typename SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::SolverType, Eigen::SparseMatrix<T>> SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::solve(const Eigen::SparseMatrix<T>& b) const {
    return this->_ldlt.solve(b);
  }

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  template<class Derived>
  const Eigen::Solve<typename SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::SolverType, Derived> SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::solve(const Eigen::MatrixBase<Derived>& b) const {
    return this->_ldlt.solve(b);
  }

  //---------------------------------------------------------------------------------------------------

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  void SolverCholeskySparse<matTypeV, T, B, NB, EigenOrderingMethod>::init(MatType& input, const VecType& b) {
    SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::init(input, b);
    _incVector.resize(b.segmentDescriptionCSPtr());
  }

  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  bool SolverCholeskySparse<matTypeV, T, B, NB, EigenOrderingMethod>::computeIncrement(MatType& input, const VecType& b, T relLambda, T absLambda) {
    return SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::computeIncrement(input, b, relLambda, absLambda, _incVector);
  }


}