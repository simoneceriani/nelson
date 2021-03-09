#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"
#include "mat/MatrixBlockTypeTraits.h"

namespace nelson {

  template<int matType, class T, int Ordering, int BR, int BC, int NBR = mat::Dynamic, int NBC = mat::Dynamic>
  class SparseWrapper {

  };


  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC> {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDense, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatOutputType = typename MatCopyType::StorageType;
    static constexpr int matOutputType = mat::BlockCoeffSparse;

  private:
    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
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

    MatCopyType& matBlocks() {
      assert(_matrix != nullptr);
      return _matCopy;
    }
    const MatCopyType& matBlocks() const {
      assert(_matrix != nullptr);
      return _matCopy;
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC> {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatOutputType = typename MatType::StorageType;
    static constexpr int matOutputType = mat::BlockCoeffSparse;
  private:


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

    MatType& matBlocks() {
      assert(_matrix != nullptr);
      return *_matrix;
    }
    const MatType& matBlocks() const {
      assert(_matrix != nullptr);
      return *_matrix;
    }

  };


  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC> {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatOutputType = typename MatCopyType::StorageType;
    static constexpr int matOutputType = mat::BlockCoeffSparse;
  private:


    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
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

    MatCopyType& matBlocks() {
      assert(_matrix != nullptr);
      return _matCopy;
    }
    const MatCopyType& matBlocks() const {
      assert(_matrix != nullptr);
      return _matCopy;
    }

  };

  template<class T, int Ordering, int BR, int BC, int NBR, int NBC>
  class SparseWrapper<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC> {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;

    using MatCopyType = typename mat::MatrixBlockIterableTypeTraits<mat::BlockCoeffSparse, T, Ordering, BR, BC, NBR, NBC>::MatrixType;
    using MatOutputType = typename MatCopyType::StorageType;
    static constexpr int matOutputType = mat::BlockCoeffSparse;
  private:


    // the original matrix
    MatType* _matrix;

    // the coeff sparse block, used as copy, internally contains a sparse matrix
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

    MatCopyType& matBlocks() {
      assert(_matrix != nullptr);
      return _matCopy;
    }
    const MatCopyType& matBlocks() const {
      assert(_matrix != nullptr);
      return _matCopy;
    }

  };

  //------------------------------------------------------------------------------------------------------------------

  template<int matType, class T, int Ordering, int BR, int NBR = mat::Dynamic>
  class SparseSquareWrapper : public SparseWrapper<matType, T, Ordering, BR, BR, NBR, NBR> {
  public:

    using MatType = typename SparseWrapper<matType, T, Ordering, BR, BR, NBR, NBR>::MatType;
    using DiagType = typename mat::VectorBlockTraits<T, BR, NBR>::StorageType;
    static constexpr int matOutputType = mat::BlockCoeffSparse;

  private:

    mutable typename mat::VectorBlockTraits<int, BR, NBR>::StorageType _diagonalCoeffIndexes;
    mutable bool _diagonalCoeffIndexesInitialized;

    // lasy search for coefficients
    void initDiagonalCoefficients() const;

  public:


    SparseSquareWrapper();
    virtual ~SparseSquareWrapper();

    bool diagonalCoeffIndexesInitialized() const {
      return _diagonalCoeffIndexesInitialized;
    }

    typename mat::VectorBlockTraits<int, BR, NBR>::StorageType diagonalCoeffIndexes() const {
      return _diagonalCoeffIndexes;
    }

    DiagType diagonal() const;
    T maxAbsDiag() const;

    template<class Derived>
    void diagonalCopy(Eigen::DenseBase<Derived> & dest) const;

    template<class Derived>
    void setDiagonal(const Eigen::DenseBase<Derived>& values);

  };

}