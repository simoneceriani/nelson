#pragma once
#include "SolverCholeskySchur.h"

#include "DoubleSectionHessianMatrices.hpp"

#include "MatrixDenseWrapper.hpp"
#include "mat/VectorBlock.hpp"

namespace nelson {


  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int solverVType,
    int wrapperWType
  >
    void SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, solverVType, wrapperWType>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b) {

    _solverVMatrix.init(input.V(), b.bV());
    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());

    _matrixW.set(&input.W());

  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int solverVType,
    int wrapperWType
  >
  T SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, solverVType, wrapperWType>::maxAbsHDiag() const
  {
    //return std::max(_solverVMatrix.maxAbsHDiag(), ;
    assert(false);
    return -1;
  }

  template<
    int matTypeU, int matTypeV, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int solverVType,
    int wrapperWType
  >
  bool SolverCholeskySchur <matTypeU, matTypeV, matTypeW, T, BU, BV, NBU, NBV, solverVType, wrapperWType>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {
    
    bool ok = true;

    // V^-1 * bV
    ok = _solverVMatrix.computeIncrement(input.V(), b.bV(), relLambda, absLambda);

    if (ok) {

      // refresh W (copy if needed)
      _matrixW.refresh();

      // bS = bU - W * V^-1 * bV
      _bS.mat() = b.bU().mat() - _matrixW.mat() * _solverVMatrix.incrementVector().mat();

      // V^-1 * Wt 
      _solverVMatrix.solve(_matrixW.mat().transpose());
      // auto LinvWt = _solverVMatrix.LinvMult(_matrixW.mat().transpose());
      // auto S2 = LinvWt.transpose() * _solverVMatrix.vectorD().cwiseInverse().asDiagonal() * LinvWt;
    }

    // TODO
    ok = false;

    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;
  }


}