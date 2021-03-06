#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

#include "ParallelExecHelper.h"

namespace nelson {

  namespace _private {
    struct UIDPair {
      int uid_1, uid_2;
    };
    struct UIDPairUS {
      int uid_U, uid_S;
    };
  }

  template<int matOutputType, class T, int BR, int NBR>
  class MatrixWWtMultiplier {
  public:
    using MatOuputType = typename mat::MatrixBlockIterableTypeTraits< matOutputType, T, mat::ColMajor, BR, BR, NBR, NBR>::MatrixType;

    using Settings = ParallelExecSettings;

  private:

    using UIDPair = _private::UIDPair;
    using UIDPairUS = _private::UIDPairUS;
    std::vector< std::vector<UIDPair> > _blockPairs;
    std::vector<UIDPairUS> _blockPairsU;
    MatOuputType _matOutput;

    Settings _settings;

  public:

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    // only one input, no need for the other matrix, sparsity pattern is the transpose
    template<class MatrixBlockU, class MatrixBlockW>
    void prepare(const MatrixBlockU& U, const MatrixBlockW & Wpattern);
    
    template<class MatrixBlockU, class MatrixBlockW>
    void multiply(
      const MatrixBlockU& U,
      const MatrixBlockW& A,
      const MatrixBlockW& B
    );

    const MatOuputType& result() const {
      return _matOutput;
    }
    MatOuputType& result() {
      return _matOutput;
    }
  };

}
