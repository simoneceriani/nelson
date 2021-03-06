#pragma once
#include "Global.h"

#include "MatrixSparseWrapper.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>	

namespace nelson {

  template<class EigenMatType, class EigenOrderingMethod>
  class SolverCholeskyEigenSparse {

    Eigen::SimplicialLDLT<EigenMatType, Eigen::Upper, EigenOrderingMethod> _ldlt;

  public:
    void init(EigenMatType& mat);

    bool factorize(EigenMatType& matInput);

    template<class Derived1, class Derived2>
    void solve(const Eigen::MatrixBase<Derived1>& b, Eigen::MatrixBase<Derived2>& x);
  };


  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  class SolverCholeskySparseBase {

  public:

    using MatTraits = mat::MatrixBlockIterableTypeTraits<matTypeV, T, mat::ColMajor, B, B, NB, NB>;
    using MatType = typename MatTraits::MatrixType;
    using VecType = mat::VectorBlock<T, B, NB>;

    using SparseWrapperT = SparseSquareWrapper<matTypeV, T, mat::ColMajor, B, NB>;
    using DiagType = typename SparseWrapperT::DiagType;

    using Type = T;

    static constexpr bool hasSettings = false;
    class Settings {};

    using SolverType = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>, Eigen::Upper, EigenOrderingMethod>;

  private:

    SparseWrapperT _sparseWrapper;
    DiagType _diagBackup;

    SolverType _ldlt;

  public:

    virtual void init(MatType& input, const VecType & b);

    T maxAbsHDiag() const;
    bool computeIncrement(MatType& input, const VecType & b, T relLambda, T absLambda, VecType& result);

    const Eigen::Solve<SolverType, Eigen::SparseMatrix<T>> solve(const Eigen::SparseMatrix<T>& b) const;

    template<class Derived>
    const Eigen::Solve<SolverType, Derived> solve(const Eigen::MatrixBase<Derived>& b) const;

  };
  
  template<int matTypeV, class T, int B, int NB, class EigenOrderingMethod>
  class SolverCholeskySparse : public SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod> {

  public:

    using MatType = typename SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::MatType;
    using VecType = typename SolverCholeskySparseBase<matTypeV, T, B, NB, EigenOrderingMethod>::VecType;

  private:

    VecType _incVector;

  public:

    void init(MatType& input, const VecType & b) override;

    bool computeIncrement(MatType& input, const VecType & b, T relLambda, T absLambda);

    const VecType& incrementVector() const {
      return _incVector;
    }

    T incrementVectorSquaredNorm() const {
      return _incVector.mat().squaredNorm();
    }

  };

}