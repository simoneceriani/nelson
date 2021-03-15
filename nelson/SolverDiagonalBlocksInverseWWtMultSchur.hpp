#pragma once
#include "SolverDiagonalBlocksInverseWWtMultSchur.h"

#include "mat/VectorBlock.hpp"

#include "MatrixDiagInv.hpp"
#include "MatrixWVinvMultiplier.hpp"
#include "MatrixWtXMultiplier.hpp"
#include "MatrixWWtMultiplier.hpp"

namespace nelson {

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType,
    int choleskyOrderingS
  >
    SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, SType, choleskyOrderingS>::SolverDiagonalBlocksInverseWWtMultSchur() :
    _settings(_Vinv.settings(), _WVinv.settings(), _wwtMult.settings(), _WtX.settings()),
    _uv_maxAbsHDiag(-1)
  {

  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType, 
    int choleskyOrderingS
  >
    void SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, SType, choleskyOrderingS>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b)
  {
    _timeStats.startInit = std::chrono::steady_clock::now();


    // prepare vectors
    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _bS.resize(b.bU().segmentDescriptionCSPtr());

    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());
    _bVtilde.resize(b.bV().segmentDescriptionCSPtr());

    // prepare V^-1 management    
    _Vinv.init(input.V());

    // prepare W*V^-1 management    
    _WVinv.prepare(input.W());

    // sparsity pattern in a sparse matrix
    auto sp = input.W().sparsityPattern().toSparseMatrix();

    // prepare W*V^-1*W' management    
    _wwtMult.prepare(input.U(), input.W(), &sp);

    // prepare the solver
    _solverS.init(_wwtMult.result(), _bS);

    // prepare W'*xu
    _WtX.prepare(input.W(), &sp);

    // temporary, not extremely smart....
    _uv_maxAbsHDiag = Eigen::NumTraits<T>::lowest();
    for (int c = 0; c < input.V().numBlocksCol(); c++) {
      c = std::max(_uv_maxAbsHDiag, input.V().blockByUID(c).diagonal().cwiseAbs().maxCoeff());
    }
    for (int c = 0; c < input.U().numBlocksCol(); c++) {
      _uv_maxAbsHDiag = std::max(_uv_maxAbsHDiag, input.U().block(c,c).diagonal().cwiseAbs().maxCoeff());
    }

    _timeStats.endInit = std::chrono::steady_clock::now();
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType,
    int choleskyOrderingS
  >
    T SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, SType, choleskyOrderingS>::maxAbsHDiag() const
  {
    return _uv_maxAbsHDiag;
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType,
    int choleskyOrderingS
  >
    bool SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, SType, choleskyOrderingS>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {
    _timeStats.addIteration();

    bool ok = true;

    // V^-1
    _timeStats.lastIteration().t0_startIteration = std::chrono::steady_clock::now();
    _Vinv.compute(input.V(), relLambda, absLambda);
    _timeStats.lastIteration().t1_VInvComputed = std::chrono::steady_clock::now();

    // W * V^-1
    _WVinv.multiply(input.W(), _Vinv.Vinv());
    _timeStats.lastIteration().t2_WVinvComputed = std::chrono::steady_clock::now();

    // bs = bU - W*V^-1*bv
    _bS.mat() = b.bU().mat();
    _WVinv.rightMultVectorSub(b.bV(), _bS);
    _timeStats.lastIteration().t3_bSComputed = std::chrono::steady_clock::now();

    // S = W * V^-1 * W'
    _wwtMult.multiply(input.U(), _WVinv.result(), input.W());
    _timeStats.lastIteration().t4_SComputed = std::chrono::steady_clock::now();

    ok = _solverS.computeIncrement(_wwtMult.result(), _bS, relLambda, absLambda, _incVector.bU());
    _timeStats.lastIteration().t5_xUSolved = std::chrono::steady_clock::now();

    _bVtilde.mat() = -b.bV().mat();
    _WtX.rightMultVectorSub(input.W(), _incVector.bU(), _bVtilde);
    _timeStats.lastIteration().t6_bVtildeComputed = std::chrono::steady_clock::now();

    _Vinv.rightMultVector(_bVtilde, _incVector.bV());
    _timeStats.lastIteration().t7_bVComputed = std::chrono::steady_clock::now();    

    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;

  }

  

}