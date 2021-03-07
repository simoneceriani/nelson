#pragma once
#include "MatrixDiagInv.h"

namespace nelson {

  template<class T, int BV, int NBV, int matTypeOut>
  void MatrixDiagInv<T, BV, NBV, matTypeOut>::init(const MatTypeV& V) {
    _Vinv.resize(V.blockDescriptor(), V.sparsityPatternCSPtr());

  }

  template<class T, int BV, int NBV, int matTypeOut>
  void MatrixDiagInv<T, BV, NBV, matTypeOut>::compute(const MatTypeV& V, T relLambda, T absLambda) {

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(V.numBlocksCol());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      // note, compute increment solve for -b, OK
      for (int c = 0; c < V.numBlocksCol(); c++) {
        assert(_Vinv.blockUID(c, c) == c);
        _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
        if (relLambda != 0 || absLambda != 0) {
          _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
          _Vinv.blockByUID(c).diagonal().array() += absLambda;
        }
        _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
      }
    }
    else {
      // static 
      if (_settings.schedule() == ParallelSchedule::schedule_static) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int c = 0; c < V.numBlocksCol(); c++) {
            assert(_Vinv.blockUID(c, c) == c);
            _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
            if (relLambda != 0 || absLambda != 0) {
              _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
              _Vinv.blockByUID(c).diagonal().array() += absLambda;
            }
            _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int c = 0; c < V.numBlocksCol(); c++) {
          assert(_Vinv.blockUID(c, c) == c);
          _Vinv.blockByUID(c) = V.blockByUID(c).template selfadjointView<Eigen::Upper>();
          if (relLambda != 0 || absLambda != 0) {
            _Vinv.blockByUID(c).diagonal().array() *= (1 + relLambda);
            _Vinv.blockByUID(c).diagonal().array() += absLambda;
          }
          _Vinv.blockByUID(c) = _Vinv.blockByUID(c).inverse().eval();
        }
      }
    }
  }
}

