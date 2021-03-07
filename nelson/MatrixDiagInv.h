#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"

#include "ParallelExecHelper.h"

namespace nelson {
  
  using MatrixDiagInvSettings =  ParallelExecSettings;

  template<class T, int BV, int NBV, int matTypeOut>
  class MatrixDiagInv {
  public:
    using MatTraitsV = mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, T, mat::ColMajor, BV, BV, NBV, NBV>;
    using MatTypeV = typename MatTraitsV::MatrixType;

    using MatTraitsVinv = mat::MatrixBlockIterableTypeTraits<matTypeOut, T, mat::ColMajor, BV, BV, NBV, NBV>;
    using MatTypeVinv = typename MatTraitsVinv::MatrixType;
  private:
    MatTypeVinv _Vinv;
    
    MatrixDiagInvSettings _settings;
  public:

    const MatrixDiagInvSettings& settings() const { return _settings; }
    MatrixDiagInvSettings& settings() { return _settings; }

    void init(const MatTypeV& V);
    void compute(const MatTypeV& V, T relLambda, T absLambda);

    const MatTypeVinv& Vinv() const { return _Vinv; }
  };


}