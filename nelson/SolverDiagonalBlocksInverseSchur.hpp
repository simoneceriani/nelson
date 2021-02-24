#pragma once
#include "SolverDiagonalBlocksInverseSchur.h"

#include "mat/VectorBlock.hpp"

namespace nelson {

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    SolverDiagonalBlocksInverseSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::SolverDiagonalBlocksInverseSchur()
    : _firstTime(true), _v_maxAbsHDiag(-1)
  {

  }


  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    void SolverDiagonalBlocksInverseSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::init(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b)
  {
    _matrixVInv.resize(input.V().blockDescriptor(), input.V().sparsityPatternCSPtr());

    _incVector.bU().resize(b.bU().segmentDescriptionCSPtr());
    _incVector.bV().resize(b.bV().segmentDescriptionCSPtr());

    _matrixW.set(&input.W());
    _matrixU.set(&input.U());

    // temporary
    _v_maxAbsHDiag = Eigen::NumTraits<T>::lowest();
    for (int c = 0; c < input.V().numBlocksCol(); c++) {
      assert(_matrixVInv.blockUID(c, c) == c);
      _v_maxAbsHDiag = std::max(_v_maxAbsHDiag, input.V().blockByUID(c).diagonal().cwiseAbs().maxCoeff());
    }

    _firstTime = true;

  }

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    T SolverDiagonalBlocksInverseSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::maxAbsHDiag() const
  {
    return std::max(_v_maxAbsHDiag, _matrixU.mat().diagonal().cwiseAbs().maxCoeff());
  }

  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    bool SolverDiagonalBlocksInverseSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::computeIncrement(DoubleSectionHessianMatricesT& input, const DoubleSectionHessianVectorsT& b, T relLambda, T absLambda)
  {

    bool ok = true;

    // V^-1
    computeVInv(input, relLambda, absLambda);

    // refresh (copy if needed)
    _matrixW.refresh();
    _matrixU.refresh();

    // bS = (-bU) - W * V^-1 * (-bV)
    // note, change sing to bU
    _bS = -b.bU().mat() - _matrixW.mat() * _matrixVInv.mat() * (-b.bV().mat());

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

    // solve!
    if (_firstTime) {
      _solverS.init(_matrixS);
      _firstTime = false;
    }

    _solverS.factorize(_matrixS);
    _solverS.solve(_bS, _incVector.bU().mat());

    // (-bV) - Wt * xu
    _bVtilde = -b.bV().mat() - _matrixW.mat().transpose() * _incVector.bU().mat();
    _incVector.bV().mat() = _matrixVInv.mat() * _bVtilde;


    if (!ok) {
      // defense
      _incVector.setZero();
    }

    return ok;
  }


  template<
    int matTypeU, int matTypeW,
    class T,
    int BU, int BV,
    int NBU, int NBV,
    int wrapperUType, int wrapperWType,
    int choleskyOrderingS
  >
    void SolverDiagonalBlocksInverseSchur<matTypeU, matTypeW, T, BU, BV, NBU, NBV, wrapperUType, wrapperWType, choleskyOrderingS>::computeVInv(DoubleSectionHessianMatricesT& input, T relLambda, T absLambda)
  {

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(input.V().numBlocksCol());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      // note, compute increment solve for -b, OK
      for (int c = 0; c < input.V().numBlocksCol(); c++) {
        assert(_matrixVInv.blockUID(c, c) == c);
        _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
        if (relLambda != 0 || absLambda != 0) {
          _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
          _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
        }
        _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
      }
    }
    else {
      // static 
      if (_settings.schedule() == ParallelSchedule::schedule_static) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int c = 0; c < input.V().numBlocksCol(); c++) {
            assert(_matrixVInv.blockUID(c, c) == c);
            _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
            }
            _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int c = 0; c < input.V().numBlocksCol(); c++) {
          assert(_matrixVInv.blockUID(c, c) == c);
          _matrixVInv.blockByUID(c) = input.V().blockByUID(c).template selfadjointView<Eigen::Upper>();
          if (relLambda != 0 || absLambda != 0) {
            _matrixVInv.blockByUID(c).diagonal().array() *= (1 + relLambda);
            _matrixVInv.blockByUID(c).diagonal().array() += absLambda;
          }
          _matrixVInv.blockByUID(c) = _matrixVInv.blockByUID(c).inverse().eval();
        }
      }
    }
  }

}