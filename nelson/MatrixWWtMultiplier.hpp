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

    _multPattern.clear();
    _multPattern.resize(input.numBlocksRow());

    std::vector<UIDPair> pairs;

    for (int i = 0; i < input.numBlocksRow(); i++) {
      // upper triag only
      for (int j = i; j < input.numBlocksRow(); j++) {
        // C[i,j] = sum_k A[i,k] * B[k,j]
        auto i_it = input.rowBegin(i);
        auto j_it = input.rowBegin(j);
        
        // TODO: reserve?
        // pairs.reserve(input.numBlocksCol());

        while (i_it() != i_it.end() && j_it() != j_it.end()) {
          int k_i = i_it.col();
          int k_j = j_it.col();
          if (k_i == k_j) {
            int buid_1 = i_it.blockUID();
            int buid_2 = j_it.blockUID();
            i_it++;
            j_it++;
            pairs.push_back(UIDPair{ buid_1, buid_2 });

          }
          else if (k_i < k_j) {
            i_it++;
          }
          else if (k_j < k_i) {
            j_it++;
          }
        }
        // found at least one block match
        if (pairs.size() > 0) {
          // add to sparsity pattern
          sp->add(i, j);

          if (matOutputOrdering == mat::ColMajor) {
            _multPattern[j].insert(MultPattern(i,pairs));
          }
          else if (matOutputOrdering == mat::RowMajor) {
            _multPattern[i].insert(MultPattern(j, pairs));
          }
          else {
            assert(false);
          }
        }


      }
    }

    _matOutput.resize(
      mat::MatrixBlockDescriptor<BR, BR, NBR, NBR>::squareMatrix(input.blockDescriptor().rowDescriptionCSPtr()),
      sp
    );

  }

  template<int matType, class T, int BR, int BC, int NBR, int NBC, int matOutputType, int matOutputOrdering>
  void MatrixWWtMultiplier<matType, T, BR, BC, NBR, NBC, matOutputType, matOutputOrdering>::multiply(const MatType& A, const MatType& B) {
    for (int o = 0; o < _multPattern.size(); o++) {
      for (const auto & it : _multPattern[o]) {
        auto block = _matOutput.blockByUID(it.inner_index); 
        block.setZero();
        for (const auto& p : it.pairs) {
          block += A.blockByUID(p.uid_1) * B.blockByUID(p.uid_1).transpose();
        }
      }
    }
  }

}
