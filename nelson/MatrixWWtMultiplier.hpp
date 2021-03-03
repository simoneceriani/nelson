#pragma once
#include "MatrixWWtMultiplier.h"

#include "mat/DenseMatrixBlock.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffMatrixBlock.hpp"
#include "mat/SparseMatrixBlock.hpp"
#include <cassert>


namespace nelson {

  template<int matType, class T, int BR, int BC, int NBR, int NBC, int matOutputType, int matOutputOrdering>
  void MatrixWWtMultiplier<matType, T, BR, BC, NBR, NBC, matOutputType, matOutputOrdering>::prepare(const MatType& input) {
    mat::SparsityPattern<matOutputOrdering>::SPtr sp(new mat::SparsityPattern<matOutputOrdering>(input.numBlocksRow(), input.numBlocksRow()));

    std::vector<std::map<int, std::vector<matrixWWtMultiplier::UIDPair>>> multPattern(input.numBlocksRow());

    std::vector<matrixWWtMultiplier::UIDPair> pairs;

    for (int i = 0; i < input.numBlocksRow(); i++) {
      // upper triag only
      for (int j = i; j < input.numBlocksRow(); j++) {
        // C[i,j] = sum_k A[i,k] * B[k,j]
        auto i_it = input.rowBegin(i);
        auto j_it = input.rowBegin(j);

        int count = 0;
        // pairs.reserve(input.numBlocksCol());
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

        pairs.resize(count);
        // redo to insert
        if (count > 0) {
          count = 0;
          i_it = input.rowBegin(i);
          j_it = input.rowBegin(j);
          while (i_it() != i_it.end() && j_it() != j_it.end()) {
            int k_i = i_it.col();
            int k_j = j_it.col();
            if (k_i == k_j) {
              int buid_1 = i_it.blockUID();
              int buid_2 = j_it.blockUID();
              i_it++;
              j_it++;
              pairs[count++] = matrixWWtMultiplier::UIDPair{ buid_1, buid_2 };

            }
            else if (k_i < k_j) {
              i_it++;
            }
            else if (k_j < k_i) {
              j_it++;
            }
          }

          assert(count == pairs.size());

          // found at least one block match
          assert(pairs.size() > 0);
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
      mat::MatrixBlockDescriptor<BR, BR, NBR, NBR>::squareMatrix(input.blockDescriptor().rowDescriptionCSPtr()),
      sp
    );
    _matOutput.setZero();

  }

  template<int matType, class T, int BR, int BC, int NBR, int NBC, int matOutputType, int matOutputOrdering>
  void MatrixWWtMultiplier<matType, T, BR, BC, NBR, NBC, matOutputType, matOutputOrdering>::multiply(const MatType& A, const MatType& B) {
    _matOutput.setZero();

    const int chunkSize = _settings.chunkSize();
    const int numEval = int(_blockPairs.size());
    const int reqNumThread = std::min(numEval, _settings.maxNumThreads());

    if (_settings.isSingleThread() || reqNumThread == 1) {
      // note, compute increment solve for -b, OK
      for (int ip = 0; ip < _blockPairs.size(); ip++) {
        typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
        
        for (const auto& p : _blockPairs[ip]) {
          block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
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
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_guided) {
        if (_settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int ip = 0; ip < _blockPairs.size(); ip++) {
            typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
            
            for (const auto& p : _blockPairs[ip]) {
              block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
            }
          }

        }
      }
      else if (_settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int ip = 0; ip < _blockPairs.size(); ip++) {
          typename MatOuputType::BlockType block = _matOutput.blockByUID(ip);
          
          for (const auto& p : _blockPairs[ip]) {
            block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_2).transpose();
          }
        }
      }
    }


  }

}
