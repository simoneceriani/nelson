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
    for (int cb = 0; cb < this->_matrix->outerSize(); cb++) {
      for (auto it = this->_matrix->begin(cb); it() != it.end(); it++) {
        //this->_matCopy.block(it.row(), it.col()) = it.block();
        this->_matCopy.blockByUID(it.blockUID()) = it.block();
      }
    }
  }

  //---------------------------------------------------------------------------------------------------

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

  }

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  SparseWrapper<mat::SparseCoeffBlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::~SparseWrapper() {

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
    for (int cb = 0; cb < this->_matrix->outerSize(); cb++) {
      for (auto it = this->_matrix->begin(cb); it() != it.end(); it++) {
        //this->_matCopy.block(it.row(), it.col()) = it.block();
        this->_matCopy.blockByUID(it.blockUID()) = it.block();
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
    for (int cb = 0; cb < this->_matrix->outerSize(); cb++) {
      for (auto it = this->_matrix->begin(cb); it() != it.end(); it++) {
        //this->_matCopy.block(it.row(), it.col()) = it.block();
        this->_matCopy.blockByUID(it.blockUID()) = it.block();
      }
    }
  }


  //---------------------------------------------------------------------------------------------------

  template<int matType, class T, int Ordering, int BR, int NBR>
  SparseSquareWrapper<matType, T, Ordering, BR, NBR>::SparseSquareWrapper()
    : SparseWrapper<matType, T, Ordering, BR, BR, NBR, NBR>(),
    _diagonalCoeffIndexesInitialized(false)
  {
    // if it is fixed size
    _diagonalCoeffIndexes.setConstant(-1);
  }


  template<int matType, class T, int Ordering, int BR, int NBR>
  SparseSquareWrapper<matType, T, Ordering, BR, NBR>::~SparseSquareWrapper() {
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  void SparseSquareWrapper<matType, T, Ordering, BR, NBR>::initDiagonalCoefficients() const {
    if (!_diagonalCoeffIndexesInitialized) {
      _diagonalCoeffIndexesInitialized = true;
      // store indexes
      assert(this->mat().rows() == this->mat().cols());
      _diagonalCoeffIndexes.resize(this->mat().rows());
      int c = 0;
      // detect diagonal blocks
      for (int o = 0; o < this->matBlocks().sparsityPattern().outerSize(); o++) {
        const auto& innerSet = this->matBlocks().sparsityPattern().inner(o);
        int inCount = innerSet.size() - 1;
        for (auto rit = innerSet.crbegin(); rit != innerSet.crend(); rit++) {
          int in = *rit;
          int i = this->matBlocks().row(o, in);
          int j = this->matBlocks().col(o, in);
          if (i == j) {
            assert(this->matBlocks().rowBlockSize(i) == this->matBlocks().colBlockSize(j));
            int offset = this->matBlocks().blockCoeffStart(o, inCount);
            int ostride = this->matBlocks().blockCoeffStride(o);
            for (int k = 0; k < this->matBlocks().rowBlockSize(i); k++) {
              _diagonalCoeffIndexes[c++] = offset + (ostride + 1) * k;
            }
            break;
          }
          inCount--;
        }
      }
      assert(c == _diagonalCoeffIndexes.size());
    }
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  typename SparseSquareWrapper<matType, T, Ordering, BR, NBR>::DiagType SparseSquareWrapper<matType, T, Ordering, BR, NBR>::diagonal() const {
    initDiagonalCoefficients(); // lazy
    SparseSquareWrapper<matType, T, Ordering, BR, NBR>::DiagType ret(_diagonalCoeffIndexes.size());
    this->diagonalCopy(ret);
    return ret;
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  T SparseSquareWrapper<matType, T, Ordering, BR, NBR>::maxAbsDiag() const {
    return diagonal().cwiseAbs().maxCoeff();
  }

  template<int matType, class T, int Ordering, int BR, int NBR>
  template<class Derived>
  void SparseSquareWrapper<matType, T, Ordering, BR, NBR>::diagonalCopy(Eigen::DenseBase<Derived>& dest) const {
    initDiagonalCoefficients(); // lazy
    assert(dest.size() == _diagonalCoeffIndexes.size());
    // in Eigen 3.4 this will be easier with slices!!
    for (int i = 0; i < _diagonalCoeffIndexes.size(); i++) {
      dest(i) = this->mat().coeffs()(_diagonalCoeffIndexes(i));
    }
  }


  template<int matType, class T, int Ordering, int BR, int NBR>
  template<class Derived>
  void SparseSquareWrapper<matType, T, Ordering, BR, NBR>::setDiagonal(const Eigen::DenseBase<Derived>& values) {
    initDiagonalCoefficients(); // lazy
    // in Eigen 3.4 this will be easier with slices!!
    assert(values.size() == _diagonalCoeffIndexes.size());
    for (int i = 0; i < _diagonalCoeffIndexes.size(); i++) {
      this->mat().coeffs()(_diagonalCoeffIndexes(i)) = values(i);
    }
  }

}

