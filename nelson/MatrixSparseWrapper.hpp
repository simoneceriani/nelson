#pragma once
#include "MatrixSparseWrapper.h"

namespace nelson {

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor(), this->_matrix->sparsityPatternCSPtr());
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->numBlocksCol(); cb++) {
      for (auto it = this->_matrix->colBegin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }

  //---------------------------------------------------------------------------------------------------

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

  }
  
  //---------------------------------------------------------------------------------------------------


  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor(), this->_matrix->sparsityPatternCSPtr());
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->numBlocksCol(); cb++) {
      for (auto it = this->_matrix->colBegin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::set(MatType* matrix) {
    this->_matrix = matrix;
    this->_matCopy.resize(this->_matrix->blockDescriptor(), this->_matrix->sparsityPatternCSPtr());
    this->refresh();
  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  void SparseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::refresh() {
    assert(this->_matrix != nullptr);
    for (int cb = 0; cb < this->_matrix->numBlocksCol(); cb++) {
      for (auto it = this->_matrix->colBegin(cb); it() != it.end(); it++) {
        this->_matCopy.block(it.row(), it.col()) = it.block();
      }
    }
  }
}