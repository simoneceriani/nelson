#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDFD : public Points2d3dBase < Points2d3dFDFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, numPoints2d, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionFDFD_DenseDiagoDense = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DenseDiagoDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseDiagoSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DenseDiagoSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DiagoDiagoDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DiagoDiagoDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoDiagoSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DiagoDiagoSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SparsDiagoDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SparsDiagoDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsDiagoSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SparsDiagoSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SpacoDiagoDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SpacoDiagoDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoDiagoSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SpacoDiagoSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDFD", "[DoubleSection-FDFD]",
  PointsSectionFDFD_DenseDiagoDense,
  PointsSectionFDFD_DenseDiagoDiago,
  PointsSectionFDFD_DenseDiagoSpars,
  PointsSectionFDFD_DenseDiagoSpaco,
  PointsSectionFDFD_DiagoDiagoDense,
  PointsSectionFDFD_DiagoDiagoDiago,
  PointsSectionFDFD_DiagoDiagoSpars,
  PointsSectionFDFD_DiagoDiagoSpaco,
  PointsSectionFDFD_SparsDiagoDense,
  PointsSectionFDFD_SparsDiagoDiago,
  PointsSectionFDFD_SparsDiagoSpars,
  PointsSectionFDFD_SparsDiagoSpaco,
  PointsSectionFDFD_SpacoDiagoDense,
  PointsSectionFDFD_SpacoDiagoDiago,
  PointsSectionFDFD_SpacoDiagoSpars,
  PointsSectionFDFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

