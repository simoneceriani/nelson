#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

namespace nelson {

  template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
  class SparseWrapper {

  };

  
  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, mat::ColMajor, BR, BC, NBR, NBC>::MatrixType;
    MatCopyType _matCopy;

  public:

    SparseWrapper() : _matrix(nullptr) {

    }

    virtual ~SparseWrapper();

    void set(MatType* matrix);

    void refresh();

    typename MatCopyType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

    const typename MatCopyType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

  public:

    SparseWrapper() : _matrix(nullptr) {

    }

    virtual ~SparseWrapper();

    void set(MatType* matrix) {
      this->_matrix = matrix;
      this->refresh();
    }

    void refresh() {
      // nothig to do
    }

    typename MatType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }

    typename MatType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }

  };

  
  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, mat::ColMajor, BR, BC, NBR, NBC>::MatrixType;
    MatCopyType _matCopy;

  public:

    SparseWrapper() : _matrix(nullptr) {

    }

    virtual ~SparseWrapper();

    void set(MatType* matrix);

    void refresh();

    typename MatCopyType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

    const typename MatCopyType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

  };
  
  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, mat::ColMajor, BR, BC, NBR, NBC>::MatrixType;
    MatCopyType _matCopy;

  public:

    SparseWrapper() : _matrix(nullptr) {

    }

    virtual ~SparseWrapper();

    void set(MatType* matrix);

    void refresh();

    typename MatCopyType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

    const typename MatCopyType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

  };
  
}