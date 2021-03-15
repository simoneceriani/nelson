#pragma once
#include "SolverCholeskySchur.h"

#include "MatrixDenseWrapper.hpp"
#include "MatrixSparseWrapper.hpp"

#include "mat/VectorBlock.hpp"

#include "DoubleSectionHessianMatrices.hpp"
#include "SingleSectionHessian.hpp"
#include "SolverCholeskyDense.hpp"
#include "SolverCholeskySparse.hpp"

namespace nelson {

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType,
    int wrapperWType,
    int solverVType,
    int choleskyOrderingS,
    int choleskyOrderingV
  >
    SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, solverVType, choleskyOrderingS, choleskyOrderingV>::SolverCholeskySchur() 
    : _firstTime(true)
  {

  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType,
    int wrapperWType,
    int solverVType,
    int choleskyOrderingS,
    int choleskyOrderingV
  >
    void SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, solverVType, choleskyOrderingS, choleskyOrderingV>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b) {

    _timeStats.startInit = std::chrono::steady_clock::now();

    _solverVMatrix.init(input.V(), b.bV());

    _timeStats.t_initVSolver = std::chrono::steady_clock::now();

    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());

    _matrixW.set(&input.W());
    _matrixU.set(&input.U());

    _firstTime = true;

    _timeStats.endInit = std::chrono::steady_clock::now();

  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType,
    int wrapperWType,
    int solverVType,
    int choleskyOrderingS,
    int choleskyOrderingV
  >
  T SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, solverVType, choleskyOrderingS, choleskyOrderingV>::maxAbsHDiag() const
  {
    return std::max(_solverVMatrix.maxAbsHDiag(), _matrixU.mat().diagonal().cwiseAbs().maxCoeff());
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType,
    int wrapperWType,
    int solverVType, 
    int choleskyOrderingS,
    int choleskyOrderingV
  >
  bool SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, solverVType, choleskyOrderingS, choleskyOrderingV>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {
    _timeStats.addIteration();
    _timeStats.lastIteration().t0_startIteration = std::chrono::steady_clock::now();

    bool ok = true;

    // V^-1 * (-bV)
    // note, compute increment solve for -b, OK
    ok = _solverVMatrix.computeIncrement(input.V(), b.bV(), relLambda, absLambda);

    _timeStats.lastIteration().t1_VInvbVComputed = std::chrono::steady_clock::now();

    if (ok) {

      // refresh (copy if needed)
      _matrixW.refresh();
      _timeStats.lastIteration().t2_WRefreshed = std::chrono::steady_clock::now();
      _matrixU.refresh();
      _timeStats.lastIteration().t3_URefreshed = std::chrono::steady_clock::now();

      // bS = (-bU) - W * V^-1 * bV
      // note, change sing to bU
      _bS = -b.bU().mat() - _matrixW.mat() * _solverVMatrix.incrementVector().mat();
      _timeStats.lastIteration().t4_bSComputed = std::chrono::steady_clock::now();

      // W * V^-1 * Wt 
      _matrixS = (_matrixU.mat() - (_matrixW.mat() * _solverVMatrix.solve(_matrixW.mat().transpose()))).template triangularView<Eigen::Upper>();

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

      ok = _solverS.factorize(_matrixS);
      _timeStats.lastIteration().t7_SFactorized = std::chrono::steady_clock::now();
    }
    if(ok) {
      _solverS.solve(_bS, _incVector.bU().mat());
      _timeStats.lastIteration().t8_bUComputed = std::chrono::steady_clock::now();

      // (-bV) - Wt * xu
      _bVtilde = -b.bV().mat() - _matrixW.mat().transpose() * _incVector.bU().mat();
      _timeStats.lastIteration().t9_bVtildeComputed = std::chrono::steady_clock::now();

      _incVector.bV().mat() = _solverVMatrix.solve(_bVtilde);
      _timeStats.lastIteration().t10_bVComputed = std::chrono::steady_clock::now();
    }

    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;
  }


}