#pragma once
#include "Global.h"

#include "MatrixSparseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>	

namespace nelson {

  template<int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SolverCholeskySparse {

  public:

    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using SparseWrapperT = SparseSquareWrapper<matTypeV, T, mat::ColMajor, B, NB>;
    using DiagType = typename SparseWrapperT::DiagType;

    using Type = T;

  private:

    SparseWrapperT _sparseWrapper;
    DiagType _diagBackup;

    VecType _incVector;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper> _ldlt;

  public:

    void init(MatType& input, const mat::VectorBlock<T, B, NB>& b);

    T maxAbsHDiag() const;
    bool computeIncrement(MatType& input, const mat::VectorBlock<T, B, NB>& b, T relLambda, T absLambda);

    const VecType& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.mat().squaredNorm();
    }

    void solve(const Eigen::SparseMatrix<T>& b) const;

  };

}