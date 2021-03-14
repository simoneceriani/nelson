#pragma once
#include "SolverDiagonalBlocksInverseSchur.h"

#include "mat/VectorBlock.hpp"

#include "MatrixDiagInv.hpp"

namespace nelson {

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    SolverDiagonalBlocksInverseSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::SolverDiagonalBlocksInverseSchur()
    : _firstTime(true), _v_maxAbsHDiag(-1), _settings(_matrixVInv.settings())
  {

  }


  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    void SolverDiagonalBlocksInverseSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b)
  {
    _timeStats.startInit = std::chrono::steady_clock::now();

    _matrixVInv.init(input.V());

    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());

    _matrixW.set(&input.W());
    _matrixU.set(&input.U());

    // temporary
    _v_maxAbsHDiag = Eigen::NumTraits<T>::lowest();
    for (int c = 0; c < input.V().numBlocksCol(); c++) {
      _v_maxAbsHDiag = std::max(_v_maxAbsHDiag, input.V().blockByUID(c).diagonal().cwiseAbs().maxCoeff());
    }

    _firstTime = true;
    _timeStats.endInit = std::chrono::steady_clock::now();
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    T SolverDiagonalBlocksInverseSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::maxAbsHDiag() const
  {
    return std::max(_v_maxAbsHDiag, _matrixU.mat().diagonal().cwiseAbs().maxCoeff());
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    bool SolverDiagonalBlocksInverseSchur<matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {
    _timeStats.addIteration();
    _timeStats.lastIteration().t0_startIteration = std::chrono::steady_clock::now();
    bool ok = true;

    // V^-1
    _matrixVInv.compute(input.V(), relLambda, absLambda);

    _timeStats.lastIteration().t1_VInvComputed = std::chrono::steady_clock::now();

    // refresh (copy if needed)
    _matrixW.refresh();
    _timeStats.lastIteration().t2_WRefreshed = std::chrono::steady_clock::now();
    _matrixU.refresh();
    _timeStats.lastIteration().t3_URefreshed = std::chrono::steady_clock::now();

    // bS = (-bU) - W * V^-1 * (-bV)
    // note, change sing to bU
    _bS = -b.bU().mat() - _matrixW.mat() * _matrixVInv.Vinv().mat() * (-b.bV().mat());
    _timeStats.lastIteration().t4_bSComputed = std::chrono::steady_clock::now();

    // W * V^-1 * Wt 
    _matrixS = (_matrixU.mat() - (_matrixW.mat() * (_matrixVInv.Vinv().mat() * _matrixW.mat().transpose()).eval()));

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

    _incVector.bV().mat() = _matrixVInv.Vinv().mat() * _bVtilde;
    _timeStats.lastIteration().t10_bVComputed = std::chrono::steady_clock::now();


    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;
  }

}