#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

namespace nelson {

  template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
  class DenseWrapper {

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC> {
  public:

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatOutputType = typename MatType::StorageType;

  private:

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

    const MatOutputType& mat() const {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }
    MatOutputType& mat() {
      assert(_matrix != nullptr);
      return _matrix->mat();
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC> {
  public:

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatCopyType = typename mat::MatrixBlockTypeTraits<mat::BlockDense, T, BR, BC, NBR, NBC>::StorageType;

    using MatOutputType = MatCopyType;

  private:
    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    MatCopyType _matCopy;

  public:

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

    void set(MatType* matrix) {
      this->_matrix = matrix;
      _matCopy.setZero(this->_matrix->mat().rows(), this->_matrix->mat().cols());
      this->refresh();
    }

    void refresh() {
      assert(this->_matrix != nullptr);
      _matCopy = MatCopyType(_matrix->mat());
    }

    const MatCopyType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy;
    }
    MatCopyType& mat() {
      assert(_matrix != nullptr);
      return _matCopy;
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC> {
  public:

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatCopyType = mat::DenseMatrixBlock<T, BR, BC, NBR, NBC>;
    using MatOutputType = typename MatCopyType::StorageType;

  private:
    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    MatCopyType _matCopy;

  public:
    
    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

    void set(MatType* matrix);

    void refresh();

    const typename MatCopyType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }
    typename MatCopyType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }


  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class DenseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC> {
  public:

    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatCopyType = mat::DenseMatrixBlock<T, BR, BC, NBR, NBC>;
    using MatOutputType = typename MatCopyType::StorageType;

  private:

    // the original matrix
    MatType* _matrix;

    // the eigen dense copy
    MatCopyType _matCopy;

  public:

    using MatOutputType = typename MatCopyType::StorageType;

    DenseWrapper() : _matrix(nullptr) {

    }

    virtual ~DenseWrapper();

    void set(MatType* matrix);

    void refresh();

    const typename MatCopyType::StorageType& mat() const {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }
    typename MatCopyType::StorageType& mat() {
      assert(_matrix != nullptr);
      return _matCopy.mat();
    }

  };

  //----------------------------------------------------------------------------------

  template<int matType, class T, int Ordering, int BR, int NBR = mat::Dynamic>
  class DenseSquareWrapper : public DenseWrapper<matType, T, Ordering, BR, BR, NBR, NBR> {
  public:

    using MatType = typename DenseWrapper<matType, T, Ordering, BR, BR, NBR, NBR>::MatType;
    using DiagType = typename mat::VectorBlockTraits<T, BR, NBR>::StorageType;

    DenseSquareWrapper();
    virtual ~DenseSquareWrapper();

    DiagType diagonal() const;
    T maxAbsDiag() const;

    template<class Derived>
    void diagonalCopy(Eigen::DenseBase<Derived>& dest) const;

    template<class Derived>
    void setDiagonal(const Eigen::DenseBase<Derived>& values);

  };


}