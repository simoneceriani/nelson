#pragma once
#include "MatrixDenseWrapper.h"

namespace nelson {

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  DenseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::~DenseWrapper() {

  }

  //---------------------------------------------------------------------------------------------------

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  DenseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::~DenseWrapper() {

  }
  //---------------------------------------------------------------------------------------------------

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  DenseWrapper < mat::SparseCoeffBlockDiagonal , T, Ordering, BR, BC, NBR, NBC > ::~DenseWrapper() {

  }

  //---------------------------------------------------------------------------------------------------


  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::~DenseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor());
    this->_matCopy.setZero();
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->outerSize(); cb++) {
      for (auto it = this->_matrix->begin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }

  //---------------------------------------------------------------------------------------------------
  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::~DenseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor());
    this->_matCopy.setZero();
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->outerSize(); cb++) {
      for (auto it = this->_matrix->begin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }

  //------------------------------------------------------------------------------------------------------------
  template<int matType, class T, int Ordering, int BR, int NBR>
  DenseSquareWrapper<matType, T, Ordering, BR, NBR>::DenseSquareWrapper() :
    DenseWrapper<matType, T, Ordering, BR, BR, NBR, NBR>()
  {

  }
  template<int matType, class T, int Ordering, int BR, int NBR>
  DenseSquareWrapper<matType, T, Ordering, BR, NBR>::~DenseSquareWrapper() {

  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  typename DenseSquareWrapper<matType, T, Ordering, BR, NBR>::DiagType DenseSquareWrapper<matType, T, Ordering, BR, NBR>::diagonal() const {
    return this->mat().diagonal();
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  T DenseSquareWrapper<matType, T, Ordering, BR, NBR>::maxAbsDiag() const {
    return this->mat().diagonal().cwiseAbs().maxCoeff();
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  template<class Derived>
  void DenseSquareWrapper<matType, T, Ordering, BR, NBR>::diagonalCopy(Eigen::DenseBase<Derived>& dest) const {
    assert(dest.size() == this->mat().diagonal().size());
    dest = this->mat().diagonal();
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  template<class Derived>
  void DenseSquareWrapper<matType, T, Ordering, BR, NBR>::setDiagonal(const Eigen::DenseBase<Derived>& values) {
    this->mat().diagonal() = values;
  }

}