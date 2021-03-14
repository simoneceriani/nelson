#pragma once
#include "Global.h"

#include "mat/MatrixTypeTraits.h"
#include "mat/VectorBlock.h"

#include "ParallelExecHelper.h"

namespace nelson {
  
  struct MatrixDiagInvSettings {
    ParallelExecSettings blockInversion;
    ParallelExecSettings rightVectorMult;
  };

  template<int matType, class T, int BV, int NBV, int matTypeOut>
  class MatrixDiagInv {
  public:
    using MatTraitsV = mat::MatrixBlockIterableTypeTraits<matType, T, mat::ColMajor, BV, BV, NBV, NBV>;
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

    MatTypeVinv& Vinv() { return _Vinv; }

    void rightMultVector(const mat::VectorBlock<T, BV, NBV>& v, mat::VectorBlock<T, BV, NBV>& res) const;
  };


}