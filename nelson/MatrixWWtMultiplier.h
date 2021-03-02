#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

#include "ParallelExecHelper.h"

namespace nelson {

  namespace matrixWWtMultiplier {
    struct UIDPair {
      int uid_1, uid_2;
    };
  }

  template<int matType, class T, int BR, int BC, int NBR, int NBC, int matOutputType, int matOutputOrdering>
  class MatrixWWtMultiplier {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<matType, T, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType;
    using MatOuputType = typename mat::MatrixBlockIterableTypeTraits< matOutputType, T, matOutputOrdering, BR, BR, NBR, NBR>::MatrixType;

    using Settings = ParallelExecSettings;

  private:

    std::vector< std::vector<matrixWWtMultiplier::UIDPair> > _blockPairs;
    MatOuputType _matOutput;

    Settings _settings;

  public:

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    void prepare(const MatType& input); // only one input, no need for the other matrix, sparsity pattern is the transpose
    void multiply(const MatType& A, const MatType& B);

    const MatOuputType& result() const {
      return _matOutput;
    }
    MatOuputType& result() {
      return _matOutput;
    }
  };

}
