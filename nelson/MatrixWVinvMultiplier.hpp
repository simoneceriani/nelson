#pragma once
#include "MatrixWVinvMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include <cassert>

namespace nelson {

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWVinvMultiplier<matWType, T, BR, BC, NBR, NBC >::prepare(const MatType& W) {
    _matOutput.resize(W.blockDescriptor(), W.sparsityPatternCSPtr());
    _matOutput.setZero();
  }

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWVinvMultiplier<matWType, T, BR, BC, NBR, NBC >::multiply(const MatType& W, const MatTypeV& Vinv) {

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(W.numBlocksRow());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      for (int r = 0; r < W.numBlocksRow(); r++) {
        for (auto it = W.rowBegin(r); it() != it.end(); it++) {
          _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
        }
      }
    }
    else {
      // static 
      if (_settings.schedule() == ParallelSchedule::schedule_static) {
        if (_settings.isChunkAuto()) {
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
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
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
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
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
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int r = 0; r < W.numBlocksRow(); r++) {
          for (auto it = W.rowBegin(r); it() != it.end(); it++) {
            _matOutput.blockByUID(it.blockUID()) = W.blockByUID(it.blockUID()) * Vinv.blockByUID(it.col());
          }
        }
      }
    }
  }

}
