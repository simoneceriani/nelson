#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "ParallelExecHelper.h"

namespace nelson {

  struct MatrixWVinvMultiplierSettings {
    ParallelExecSettings multiplication;
    ParallelExecSettings rightVectorMult;
  };

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  class MatrixWVinvMultiplier {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<matWType, T, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType;
    using MatTypeV = typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, mat::ColMajor, BC, BC, NBC, NBC>::MatrixType;

    using Settings = MatrixWVinvMultiplierSettings;

  private:
    
    MatType _matOutput;

    Settings _settings;

  public:

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    // only one input, no need for the other matrix since it is the diagonal block
    void prepare(const MatType & W);

    void multiply(const MatType& W, const MatTypeV& Vinv);

    const MatType& result() const {
      return _matOutput;
    }
    MatType& result() {
      return _matOutput;
    }

    void rightMultVectorSub(const mat::VectorBlock<T, BC, NBC>& v, mat::VectorBlock<T, BR, NBR>& res) const;

  };

}