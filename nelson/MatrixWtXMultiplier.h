#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "ParallelExecHelper.h"

#include <Eigen/Sparse>
#include <vector>

namespace nelson {

  namespace _private {
    struct UIDColPair {
      int uid;
      int xid;
    };
  }

  using MatrixWtXMultiplierSettings = ParallelExecSettings;

  template<int matWType, class T, int BR, int BC, int NBR, int NBC>
  class MatrixWtXMultiplier {
  public:
    using MatType = typename mat::MatrixBlockIterableTypeTraits<matWType, T, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType;

    using Settings = MatrixWtXMultiplierSettings;

  private:

    Settings _settings;

    using UIDColPair = _private::UIDColPair;
    std::vector<std::vector<UIDColPair>> _blocks;

  public:

    Settings& settings() { return _settings; }
    const Settings& settings() const { return _settings; }

    void prepare(const MatType& W, const Eigen::SparseMatrix<int, Eigen::RowMajor>* spWmat = nullptr);

    void rightMultVectorSub(const MatType& W, const mat::VectorBlock<T, BR, NBR>& v, mat::VectorBlock<T, BC, NBC>& res) const;

  };


}