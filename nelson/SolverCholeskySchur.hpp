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

    _solverVMatrix.init(input.V(), b.bV());
    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());

    _matrixW.set(&input.W());
    _matrixU.set(&input.U());

    _firstTime = true;

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
    
    bool ok = true;

    // V^-1 * (-bV)
    // note, compute increment solve for -b, OK
    ok = _solverVMatrix.computeIncrement(input.V(), b.bV(), relLambda, absLambda);

    if (ok) {

      // refresh (copy if needed)
      _matrixW.refresh();
      _matrixU.refresh();

      // bS = (-bU) - W * V^-1 * bV
      // note, change sing to bU
      _bS = -b.bU().mat() - _matrixW.mat() * _solverVMatrix.incrementVector().mat();

      // W * V^-1 * Wt 
      _matrixS = (_matrixU.mat() - (_matrixW.mat() * _solverVMatrix.solve(_matrixW.mat().transpose()))).template triangularView<Eigen::Upper>();

      // add diagonal lambdas
      if (relLambda != 0 || absLambda != 0) {
        _matrixS.diagonal() += (relLambda * _matrixU.mat().diagonal());
        _matrixS.diagonal().array() += absLambda;
      }

      // solve!
      if (_firstTime) {
        _solverS.init(_matrixS);
        _firstTime = false;
      }

      ok = _solverS.factorize(_matrixS);
    }
    if(ok) {
      _solverS.solve(_bS, _incVector.bU().mat());

      // (-bV) - Wt * xu
      _bVtilde = -b.bV().mat() - _matrixW.mat().transpose() * _incVector.bU().mat();
      _incVector.bV().mat() = _solverVMatrix.solve(_bVtilde);

    }

    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;
  }


}