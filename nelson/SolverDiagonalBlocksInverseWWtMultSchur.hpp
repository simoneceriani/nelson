#pragma once
#include "SolverDiagonalBlocksInverseWWtMultSchur.h"

#include "mat/VectorBlock.hpp"

#include "MatrixDiagInv.hpp"
#include "MatrixWVinvMultiplier.hpp"
#include "MatrixWtXMultiplier.hpp"
#include "MatrixWWtMultiplier.hpp"

namespace nelson {

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType, int VinvType,
    int choleskyOrderingS
  >
    SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, SType, VinvType, choleskyOrderingS>::SolverDiagonalBlocksInverseWWtMultSchur() :
    _settings(_Vinv.settings(), _WVinv.settings(), _wwtMult.settings(), _WtX.settings()),
    _uv_maxAbsHDiag(-1)
  {

  }

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType, int VinvType,
    int choleskyOrderingS
  >
    void SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, SType, VinvType, choleskyOrderingS>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b)
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
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType, int VinvType,
    int choleskyOrderingS
  >
    T SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, SType, VinvType, choleskyOrderingS>::maxAbsHDiag() const
  {
    return _uv_maxAbsHDiag;
  }

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int SType, int VinvType,
    int choleskyOrderingS
  >
    bool SolverDiagonalBlocksInverseWWtMultSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, SType, VinvType, choleskyOrderingS>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {
    _timeStats.addIteration();

    bool ok = true;

    // V^-1
    _timeStats.lastIteration().t0_startIteration = std::chrono::steady_clock::now();
    _Vinv.compute(input.V(), relLambda, absLambda);
    _timeStats.lastIteration().t1_VInvComputed = std::chrono::steady_clock::now();

    // W * V^-1
    _WVinv.multiply(input.W(), _Vinv.Vinv());
    _timeStats.lastIteration().t2_VinvWComputed = std::chrono::steady_clock::now();

    // bs = bU - W*V^-1*bv
    _bS.mat() = b.bU().mat();
    _WVinv.rightMultVectorSub(b.bV(), _bS);

    // S = W * V^-1 * W'
    _wwtMult.multiply(input.U(), _WVinv.result(), input.W());

    ok = _solverS.computeIncrement(_wwtMult.result(), _bS, relLambda, absLambda);

    _bVtilde.mat() = -b.bV().mat();
    _WtX.rightMultVectorSub(input.W(), _solverS.incrementVector(), _bVtilde);

    _Vinv.rightMultVector(_bVtilde, _incVector.bV());
    _incVector.bU().mat() = _solverS.incrementVector().mat();

    if (!ok) {
      // defense
      _incVector.setZero();
    }

    /*
    // refresh (copy if needed)
    _matrixW.refresh();
    _timeStats.lastIteration().t2_WRefreshed = std::chrono::steady_clock::now();
    _matrixU.refresh();
    _timeStats.lastIteration().t3_URefreshed= std::chrono::steady_clock::now();

    // bS = (-bU) - W * V^-1 * (-bV)
    // note, change sing to bU
    _bS = -b.bU().mat() - _matrixW.mat() * _matrixVInv.mat() * (-b.bV().mat());
    _timeStats.lastIteration().t4_bSComputed = std::chrono::steady_clock::now();

    // W * V^-1 * Wt
    //_matrixS = _matrixU.mat().template triangularView<Eigen::Upper>();
    //MatrixUType::MatOutputType S2 = (_matrixW.mat() * _matrixVInv.mat() * _matrixW.mat().transpose());
    //_matrixS -= S2;
    //_matrixS = (_matrixU.mat() - (_matrixW.mat() * _matrixVInv.mat() * _matrixW.mat().transpose())).template triangularView<Eigen::Upper>();
    //_matrixS = (_matrixU.mat() - (_matrixW.mat() * _matrixVInv.mat() * _matrixW.mat().transpose()));
    _matrixS = (_matrixU.mat() - (_matrixW.mat() * (_matrixVInv.mat() * _matrixW.mat().transpose()).eval()));

    // add diagonal lambdas
    if (relLambda != 0 || absLambda != 0) {
      _matrixS.diagonal() += (relLambda * _matrixU.mat().diagonal());
      _matrixS.diagonal().array() += absLambda;
    }
    _timeStats.lastIteration().t5_SComputed = std::chrono::steady_clock::now();

    // solve!
    if (_firstTime) {
      _solverS.init(_matrixS);
      _firstTime = false;
    }
    _timeStats.lastIteration().t6_SSolveInit = std::chrono::steady_clock::now();

    _solverS.factorize(_matrixS);
    _timeStats.lastIteration().t7_SFactorized = std::chrono::steady_clock::now();

    _solverS.solve(_bS, _incVector.bU().mat());
    _timeStats.lastIteration().t8_bUComputed = std::chrono::steady_clock::now();

    // (-bV) - Wt * xu
    _bVtilde = -b.bV().mat() - _matrixW.mat().transpose() * _incVector.bU().mat();
    _timeStats.lastIteration().t9_bVtildeComputed = std::chrono::steady_clock::now();

    _incVector.bV().mat() = _matrixVInv.mat() * _bVtilde;
    _timeStats.lastIteration().t10_bVComputed = std::chrono::steady_clock::now();


    if (!ok) {
      // defense
      _incVector.setZero();
    }

    
    */
    return ok;

  }

  

}