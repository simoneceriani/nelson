#pragma once
#include "MatrixWWtMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include <cassert>


namespace nelson {

  template<int matOutputType, class T, int matOutputOrdering, int BR, int NBR>
  template<class MatrixBlockU, class MatrixBlockW>
  void MatrixWWtMultiplier<matOutputType, T, matOutputOrdering, BR, NBR>::prepare(
    const MatrixBlockU& U,
    const MatrixBlockW& W
  ) {
    assert(U.numBlocksRow() == U.numBlocksCol());
    assert(U.numBlocksRow() == W.numBlocksRow());
    mat::SparsityPattern<matOutputOrdering>::SPtr sp(new mat::SparsityPattern<matOutputOrdering>(U.numBlocksRow(), U.numBlocksRow()));

    std::vector<std::map<int, std::vector<UIDPair>>> multPattern(W.numBlocksRow());


    // prepare iterators for each column of U matrix
    std::vector<typename MatrixBlockU::InnerIterator<const MatrixBlockU>> u_j_its;
    u_j_its.reserve(U.numBlocksCol());
    for (int j = 0; j < W.numBlocksRow(); j++) {
      u_j_its.push_back(U.colBegin(j));
    }

    // iterate on rows of W
    for (int i = 0; i < W.numBlocksRow(); i++) {
      // iterate on cols of W', i.e., on rows of W, upper triag only
      for (int j = i; j < W.numBlocksRow(); j++) {
        // C[i,j] = U[i,j] - sum_k A[i,k] * B[k,j]'
        auto i_it = W.rowBegin(i);
        auto j_it = W.rowBegin(j);

        // count how many elements are both non zero
        int count = 0;
        while (i_it() != i_it.end() && j_it() != j_it.end()) {
          int k_i = i_it.col();
          int k_j = j_it.col();
          if (k_i == k_j) {
            i_it++;
            j_it++;
            count++;
          }
          else if (k_i < k_j) {
            i_it++;
          }
          else if (k_j < k_i) {
            j_it++;
          }
        }

        // check if there is an additional element from U[i,j]
        int u_ij_buid = -1;
        while (u_j_its[j]() != u_j_its[j].end() && u_j_its[j].row() < i) {
          u_j_its[j]++;
        }
        if (u_j_its[j]() != u_j_its[j].end() && u_j_its[j].row() == i) {
          u_ij_buid = u_j_its[j].blockUID();
        }

        std::vector<UIDPair> pairs(count + (u_ij_buid != -1 ? 1 : 0));
        if (u_ij_buid != -1) {
          // set uid of u and -1 as second uid, to remember this is not from W*W' but from U
          pairs[0].uid_1 = u_ij_buid;
          pairs[0].uid_2 = -1;
        }

        // redo the loop to insert
        if (count > 0) {

          count = (u_ij_buid != -1 ? 1 : 0);
          i_it = W.rowBegin(i);
          j_it = W.rowBegin(j);
          while (i_it() != i_it.end() && j_it() != j_it.end()) {
            int k_i = i_it.col();
            int k_j = j_it.col();
            if (k_i == k_j) {
              int buid_1 = i_it.blockUID();
              int buid_2 = j_it.blockUID();
              i_it++;
              j_it++;
              pairs[count++] = UIDPair{ buid_1, buid_2 };

            }
            else if (k_i < k_j) {
              i_it++;
            }
            else if (k_j < k_i) {
              j_it++;
            }
          }

          assert(count == pairs.size());
        }
        // found at least one block match (from U or W*W')
        if (pairs.size() > 0) {
          // add to sparsity pattern
          sp->add(i, j);

          if (matOutputOrdering == mat::ColMajor) {
            // MultPattern constructor will perform swap
            multPattern[j][i].swap(pairs);
          }
          else if (matOutputOrdering == mat::RowMajor) {
            // MultPattern constructor will perform swap
            multPattern[i][j].swap(pairs);
          }
          else {
            assert(false);
          }




        }
      }
    }

    _blockPairs.resize(sp->count());
    int count = 0;
    for (int o = 0; o < multPattern.size(); o++) {
      for (auto& it : multPattern[o]) {
        _blockPairs[count++].swap(it.second);
      }
    }

    _matOutput.resize(
      mat::MatrixBlockDescriptor<BR, BR, NBR, NBR>::squareMatrix(W.blockDescriptor().rowDescriptionCSPtr()),
      sp
    );
    _matOutput.setZero();

  }

  template<int matOutputType, class T, int matOutputOrdering, int BR, int NBR>
  template<class MatrixBlockU, class MatrixBlockW>
  void MatrixWWtMultiplier<matOutputType, T, matOutputOrdering, BR, NBR>::multiply(
    const MatrixBlockU& U,
    const MatrixBlockW& A,
    const MatrixBlockW& B
  )
  {
    _matOutput.setZero();

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(_blockPairs.size());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      for (int ip = 0; ip < _blockPairs.size(); ip++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
        assert(_blockPairs[ip].size() > 0); // if exists is not empty
        int i = 0;
        if (_blockPairs[ip][i].uid_2 == -1) {
          block = U.blockByUID(_blockPairs[ip][i].uid_1);
          i++;
        }
        for (; i < _blockPairs[ip].size(); i++) {
          const auto& p = _blockPairs[ip][i];
          block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
        }
      }
    }
    else {
      // static 
      if (_settings.schedule() == ParallelSchedule::schedule_static) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() > 0); // if exists is not empty
            int i = 0;
            if (_blockPairs[ip][i].uid_2 == -1) {
              block = U.blockByUID(_blockPairs[ip][i].uid_1);
              i++;
            }
            for (; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int ip = 0; ip < _blockPairs.size(); ip++) {
          typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
          assert(_blockPairs[ip].size() > 0); // if exists is not empty
          int i = 0;
          if (_blockPairs[ip][i].uid_2 == -1) {
            block = U.blockByUID(_blockPairs[ip][i].uid_1);
            i++;
          }
          for (; i < _blockPairs[ip].size(); i++) {
            const auto& p = _blockPairs[ip][i];
            block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
          }
        }
      }
    }


  }

}
