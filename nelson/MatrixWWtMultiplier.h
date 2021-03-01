#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

namespace nelson {

  template<int matType, class T, int BR, int BC, int NBR, int NBC, int matOutputType, int matOutputOrdering>
  class MatrixWWtMultiplier {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<matType, T, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType;
    using MatOuputType = typename mat::MatrixBlockIterableTypeTraits< matOutputType, T, matOutputOrdering, BR, BR, NBR, NBR>::MatrixType;
  
  private:
    struct UIDPair {
      int uid_1, uid_2;
    };

    struct MultPattern {
      int inner_index;
      std::vector<UIDPair> pairs;

      MultPattern(int inner_index, std::vector<UIDPair> & pairs) : inner_index(inner_index) {
        this->pairs.swap(pairs);
      }
    };

    struct MultPatternCmp {
      bool operator() (const MultPattern & a, const MultPattern& b) const {
        return a.inner_index < b.inner_index;
      }
    };

    std::vector<std::set<MultPattern, MultPatternCmp>> _multPattern;
    MatOuputType _matOutput;

  public:


    void prepare(const MatType& input); // only one input, no need for the other matrix, sparsity pattern is the transpose
    void multiply(const MatType& A, const MatType& B);
  };
  
}
