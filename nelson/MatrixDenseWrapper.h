#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

namespace nelson {

  template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
  class DenseWrapper {

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    MatType* _matrix;

  public:

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

    void set(MatType* matrix) {
      this->_matrix = matrix;
    }

    void refresh() {
      // no need, direct pointing to matrix
    }

    typename MatType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }

    const typename MatType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    using MatCopyType = typename mat::MatrixBlockTypeTraits<mat::BlockDense, T, BR, BC, NBR, NBC>::StorageType;
    MatCopyType _matCopy;

  public:

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

    void set(MatType* matrix) {
      this->_matrix = matrix;
      this->refresh();
    }

    void refresh() {
      assert(this->_matrix != nullptr);
      _matCopy = MatCopyType(_matrix->mat());
    }

    MatCopyType& mat() {
      assert(_matrix != nullptr);
      return _matCopy;
    }

    const MatCopyType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy;
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    using MatCopyType = mat::DenseMatrixBlock<T, BR, BC, NBR, NBC>;
    MatCopyType _matCopy;

  public:

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

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
  class DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC> {

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    using MatCopyType = mat::DenseMatrixBlock<T, BR, BC, NBR, NBC>;
    MatCopyType _matCopy;

  public:

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

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