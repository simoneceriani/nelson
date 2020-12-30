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
  DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::~DenseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor());
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->numBlocksCol(); cb++) {
      for (auto it = this->_matrix->colBegin(cb); it() != it.end(); it++) {
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
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->numBlocksCol(); cb++) {
      for (auto it = this->_matrix->colBegin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }
}