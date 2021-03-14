#pragma once
#include "MatrixDiagInv.h"
#include "mat/VectorBlock.hpp"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include "mat/SparseCoeffDiagonalMatrixBlock.hpp"

#include <cassert>

#include <Eigen/Dense>


namespace nelson {

  template<int matType, class T, int BV, int NBV, int matTypeOut>
  void MatrixDiagInv<matType, T, BV, NBV, matTypeOut>::init(const MatTypeV& V) {
    _Vinv.resize(V.blockDescriptor(), V.sparsityPatternCSPtr());

  }

  template<int matType, class T, int BV, int NBV, int matTypeOut>
  void MatrixDiagInv<matType, T, BV, NBV, matTypeOut>::compute(const MatTypeV& V, T relLambda, T absLambda) {

    const auto& settings = _settings.blockInversion;
    const int chunkSize = settings.chunkSize();
    const int numEval = int(V.numBlocksCol());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
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
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
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
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
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
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
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
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
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

  template<int matType, class T, int BV, int NBV, int matTypeOut>
  void MatrixDiagInv<matType, T, BV, NBV, matTypeOut>::rightMultVector(const mat::VectorBlock<T, BV, NBV>& v, mat::VectorBlock<T, BV, NBV>& res) const {

    const auto& settings = _settings.rightVectorMult;
    const int chunkSize = settings.chunkSize();
    const int numEval = int(_Vinv.numBlocksCol());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    assert(_Vinv.numBlocksCol() == v.numSegments());
    assert(_Vinv.numBlocksCol() == res.numSegments());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
        res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
      }
    }
    else {
      // static 
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
            res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int c = 0; c < _Vinv.numBlocksCol(); c++) {
          res.segment(c) = _Vinv.blockByUID(c) * v.segment(c);
        }
      }
    }
  }

}

