#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFDF : public Points2d3dBase < Points2d3dFFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, numPoints3d > {

public:

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFFDF_DenseDiagoDense = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DenseDiagoDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseDiagoSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DenseDiagoSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DiagoDiagoDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DiagoDiagoDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoDiagoSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DiagoDiagoSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SparsDiagoDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SparsDiagoDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsDiagoSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SparsDiagoSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SpacoDiagoDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SpacoDiagoDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoDiagoSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SpacoDiagoSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFDF", "[DoubleSection-FFDF]",
  PointsSectionFFDF_DenseDiagoDense,
  PointsSectionFFDF_DenseDiagoDiago,
  PointsSectionFFDF_DenseDiagoSpars,
  PointsSectionFFDF_DenseDiagoSpaco,
  PointsSectionFFDF_DiagoDiagoDense,
  PointsSectionFFDF_DiagoDiagoDiago,
  PointsSectionFFDF_DiagoDiagoSpars,
  PointsSectionFFDF_DiagoDiagoSpaco,
  PointsSectionFFDF_SparsDiagoDense,
  PointsSectionFFDF_SparsDiagoDiago,
  PointsSectionFFDF_SparsDiagoSpars,
  PointsSectionFFDF_SparsDiagoSpaco,
  PointsSectionFFDF_SpacoDiagoDense,
  PointsSectionFFDF_SpacoDiagoDiago,
  PointsSectionFFDF_SpacoDiagoSpars,
  PointsSectionFFDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}