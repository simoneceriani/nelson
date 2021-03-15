#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFFD : public Points2d3dBase < Points2d3dFFFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, numPoints2d, mat::Dynamic > {

public:

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionFFFD_DenseDiagoDense = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DenseDiagoDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseDiagoSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DenseDiagoSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DiagoDiagoDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DiagoDiagoDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoDiagoSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DiagoDiagoSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SparsDiagoDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SparsDiagoDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsDiagoSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SparsDiagoSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SpacoDiagoDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SpacoDiagoDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoDiagoSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SpacoDiagoSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFFD", "[DoubleSection-FFFD]",
  PointsSectionFFFD_DenseDiagoDense,
  PointsSectionFFFD_DenseDiagoDiago,
  PointsSectionFFFD_DenseDiagoSpars,
  PointsSectionFFFD_DenseDiagoSpaco,
  PointsSectionFFFD_DiagoDiagoDense,
  PointsSectionFFFD_DiagoDiagoDiago,
  PointsSectionFFFD_DiagoDiagoSpars,
  PointsSectionFFFD_DiagoDiagoSpaco,
  PointsSectionFFFD_SparsDiagoDense,
  PointsSectionFFFD_SparsDiagoDiago,
  PointsSectionFFFD_SparsDiagoSpars,
  PointsSectionFFFD_SparsDiagoSpaco,
  PointsSectionFFFD_SpacoDiagoDense,
  PointsSectionFFFD_SpacoDiagoDiago,
  PointsSectionFFFD_SpacoDiagoSpars,
  PointsSectionFFFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}