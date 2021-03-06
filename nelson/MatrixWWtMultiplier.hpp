#pragma once
#include "MatrixWWtMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include <cassert>


namespace nelson {

  namespace _private {

    template<class Iter1, class Iter2>
    void addNNZ_buid(Iter1& i_it, Iter2& j_it, std::vector<UIDPair>& pairs) {
      int count = 0;
      while (i_it() != i_it.end() && j_it() != j_it.end() && count < pairs.size()) {
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

  }

  template<int matOutputType, class T, int BR, int NBR>
  template<class MatrixBlockU, class MatrixBlockW>
  void MatrixWWtMultiplier<matOutputType, T, BR, NBR>::prepare(
    const MatrixBlockU& U,
    const MatrixBlockW& W
  ) {
    assert(U.numBlocksRow() == U.numBlocksCol());
    assert(U.numBlocksRow() == W.numBlocksRow());

    // sparsity pattern of destionation matrix
    mat::SparsityPattern<mat::ColMajor>::SPtr sp(new mat::SparsityPattern<mat::ColMajor>(U.numBlocksRow(), U.numBlocksRow()));

    // prepare the count of elements to be multiplied for the output matrix (including U blocks to be summed)
    auto spMatW = W.sparsityPattern().toSparseMatrix();
    auto spMatU = U.sparsityPattern().toSparseMatrix();
    Eigen::SparseMatrix<int> spMatUWWt = (spMatU + spMatW * spMatW.transpose()).triangularView<Eigen::Upper>();

    // iterate on columns
    int o_buid = 0;
    int u_buid = 0;
    
    // contains all blocks of output matrix ordered as they are ordered in the matrix, 
    //   per each, contains the list of blocks to be multiplied 
    _blockPairs.resize(spMatUWWt.nonZeros());

    // contains only the blocks of U Matrix, per each the UID of source and UID of destination
    _blockPairsU.resize(spMatU.nonZeros());
    for (int j = 0; j < spMatUWWt.outerSize(); ++j) {

      // iterator on U rows given j col
      auto iU_it = U.colBegin(j);

      // iterate on non empty rows
      for (Eigen::SparseMatrix<int>::InnerIterator it(spMatUWWt, j); it; ++it)
      {
        // set the sparsity pattern
        sp->add(it.row(), it.col());

        // remember row of destionation
        int i = it.row();

        int blockCounts = it.value();

        // check if there is an additional element from U[i,j]
        if (iU_it() != iU_it.end() && iU_it.row() == i) {
          _blockPairsU[u_buid].uid_U = iU_it.blockUID();
          _blockPairsU[u_buid].uid_S = o_buid;
          iU_it++;
          u_buid++;
          blockCounts--;
        }

        if (blockCounts > 0) {
          // prepare list of blocks k: S[i,j] = U[i,j] - sum_k A[i,k] * B[k,j]'
          auto i_it = W.rowBegin(i);
          auto j_it = W.rowBegin(j);
          _blockPairs[o_buid].resize(blockCounts);
          addNNZ_buid(i_it, j_it, _blockPairs[o_buid]);
        }

        // next block UID of output matrix
        o_buid++;
      }
    }
    assert(o_buid == _blockPairs.size());

    _matOutput.resize(
      mat::MatrixBlockDescriptor<BR, BR, NBR, NBR>::squareMatrix(W.blockDescriptor().rowDescriptionCSPtr()),
      sp
    );
  }

  template<int matOutputType, class T, int BR, int NBR>
  template<class MatrixBlockU, class MatrixBlockW>
  void MatrixWWtMultiplier<matOutputType, T, BR, NBR>::multiply(
    const MatrixBlockU& U,
    const MatrixBlockW& A,
    const MatrixBlockW& B
  )
  {
    _matOutput.setZero();

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(_blockPairs.size());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());
    
    const int numEvalU = int(_blockPairsU.size());
    const int reqNumThreadU = std::min(numEvalU, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      // first copy U blocks
      for (int i = 0; i < _blockPairsU.size(); i++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
        block = U.blockByUID(_blockPairsU[i].uid_U);
      }
      // then subtract W*W' blocks
      for (int ip = 0; ip < _blockPairs.size(); ip++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
        assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

        for (int i = 0; i < _blockPairs[ip].size(); i++) {
          const auto& p = _blockPairs[ip][i];
          block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
        }
      }
    }
    else {
      // static 
      if (_settings.schedule() == ParallelSchedule::schedule_static) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(static)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(static, chunkSize)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }


        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(dynamic)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(dynamic, chunkSize)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(guided)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
        else {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(guided, chunkSize)
          // first copy U blocks
          for (int i = 0; i < _blockPairsU.size(); i++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
            block = U.blockByUID(_blockPairsU[i].uid_U);
          }
          // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

            for (int i = 0; i < _blockPairs[ip].size(); i++) {
              const auto& p = _blockPairs[ip][i];
              block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThreadU) default (shared) schedule(runtime)
      // first copy U blocks
      for (int i = 0; i < _blockPairsU.size(); i++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(_blockPairsU[i].uid_S);
        block = U.blockByUID(_blockPairsU[i].uid_U);
      }
      // then subtract W*W' blocks
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
      for (int ip = 0; ip < _blockPairs.size(); ip++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
        assert(_blockPairs[ip].size() >= 0); // if exists is not empty or it is 0 if only U block exists

        for (int i = 0; i < _blockPairs[ip].size(); i++) {
          const auto& p = _blockPairs[ip][i];
          block -= A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
        }
      }

      }
    }


  }

}
