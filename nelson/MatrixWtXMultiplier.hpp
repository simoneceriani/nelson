#pragma once
#include "MatrixWtXMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include "mat/SparseCoeffDiagonalMatrixBlock.hpp"

#include "mat/VectorBlock.hpp"

#include <cassert>

namespace nelson {

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWtXMultiplier<matWType, T, BR, BC, NBR, NBC >::prepare(const MatType& W, const Eigen::SparseMatrix<int, Eigen::RowMajor>* spWmat) {
    Eigen::SparseMatrix<int, Eigen::RowMajor> spMat_compute;
    if (spWmat == nullptr) {
      spMat_compute = W.sparsityPattern().toSparseMatrix();
      spWmat = &spMat_compute;
    }
    const Eigen::SparseMatrix<int, Eigen::RowMajor>& spMat = *spWmat;
    
    Eigen::Matrix<int, 1, NBC> counts = Eigen::Matrix<int, 1, NBC>::Zero(W.numBlocksCol());
    for (int r = 0; r < spMat.outerSize(); ++r) {
      for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(spMat, r); it; ++it) {
        counts(it.col())++;
      }
    }

    _blocks.resize(W.numBlocksCol());
    for (int i = 0; i < _blocks.size(); ++i) {
      _blocks[i].resize(counts(i));
    }

    counts.setZero();
    for (int r = 0; r < W.numBlocksRow(); ++r) {
      for (auto it = W.rowBegin(r); it() != it.end(); it++) {
        _blocks[it.col()][counts(it.col())].uid = it.blockUID();
        _blocks[it.col()][counts(it.col())].xid = r;
        counts(it.col())++;
      }
    }

  }

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  void MatrixWtXMultiplier<matWType, T, BR, BC, NBR, NBC >::rightMultVectorSub(const MatType& W, const mat::VectorBlock<T, BR, NBR>& v, mat::VectorBlock<T, BC, NBC>& res) const {

    const auto& settings = _settings;

    const int chunkSize = settings.chunkSize();
    const int numEval = int(W.numBlocksRow());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (int i = 0; i < _blocks.size(); ++i) {
        for (int j = 0; j < _blocks[i].size(); ++j) {
          res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
        }
      }
    }
    else {
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int i = 0; i < _blocks.size(); ++i) {
            for (int j = 0; j < _blocks[i].size(); ++j) {
              res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
            }
          }
        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int i = 0; i < _blocks.size(); ++i) {
          for (int j = 0; j < _blocks[i].size(); ++j) {
            res.segment(i) -= W.blockByUID(_blocks[i][j].uid).transpose() * v.segment(_blocks[i][j].xid);
          }
        }
      }
    }
  }


}
