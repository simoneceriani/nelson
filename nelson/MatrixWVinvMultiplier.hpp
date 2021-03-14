#pragma once
#include "MatrixWVinvMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include "mat/SparseCoeffDiagonalMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

#include <cassert>

namespace nelson {

  template<int matWType, int matVType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWVinvMultiplier<matWType, matVType, T, BR, BC, NBR, NBC >::prepare(const MatType& W) {
    _matOutput.resize(W.blockDescriptor(), W.sparsityPatternCSPtr());
    _matOutput.setZero();
  }

  template<int matWType, int matVType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWVinvMultiplier<matWType, matVType, T, BR, BC, NBR, NBC >::multiply(const MatType& W, const MatTypeV& Vinv) {

    const auto& settings = _settings.multiplication;

    const int chunkSize = settings.chunkSize();
    const int numEval = int(W.numBlocksRow());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (int r = 0; r < W.numBlocksRow(); r++) {
        for (auto it = W.rowBegin(r); it() != it.end(); it++) {
          _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
        }
      }
    }
    else {
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int r = 0; r < W.numBlocksRow(); r++) {
            for (auto it = W.rowBegin(r); it() != it.end(); it++) {
              _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int r = 0; r < W.numBlocksRow(); r++) {
          for (auto it = W.rowBegin(r); it() != it.end(); it++) {
            _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
          }
        }
      }
    }
  }


  template<int matWType, int matVType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWVinvMultiplier<matWType, matVType, T, BR, BC, NBR, NBC >::rightMultVectorSub(const mat::VectorBlock<T, BC, NBC>& v, mat::VectorBlock<T, BR, NBR>& res) const {
    const auto& settings = _settings.rightVectorMult;

    const int chunkSize = settings.chunkSize();
    const int numEval = int(_matOutput.numBlocksRow());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
        for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
          res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
        }
      }
    }
    else {
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
            for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
              res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int r = 0; r < _matOutput.numBlocksRow(); r++) {
          for (auto it = _matOutput.rowBegin(r); it() != it.end(); it++) {
            res.segment(r) -= _matOutput.blockByUID(it.blockUID()) * v.segment(it.col());
          }
        }
      }
    }

  }

}
