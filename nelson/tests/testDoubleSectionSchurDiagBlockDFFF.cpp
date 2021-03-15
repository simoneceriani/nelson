#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFFF : public Points2d3dBase < Points2d3dDFFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, numPoints2d, numPoints3d > {

public:
  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }


};

using PointsSectionDFFF_DenseDiagoDense = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_DenseDiagoDiago = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_DenseDiagoSpars = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_DenseDiagoSpaco = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_DiagoDiagoDense = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_DiagoDiagoDiago = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_DiagoDiagoSpars = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_DiagoDiagoSpaco = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_SparsDiagoDense = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_SparsDiagoDiago = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_SparsDiagoSpars = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_SparsDiagoSpaco = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_SpacoDiagoDense = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_SpacoDiagoDiago = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_SpacoDiagoSpars = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_SpacoDiagoSpaco = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DFFF", "[DoubleSection-DFFF]",
  PointsSectionDFFF_DenseDiagoDense,
  PointsSectionDFFF_DenseDiagoDiago,
  PointsSectionDFFF_DenseDiagoSpars,
  PointsSectionDFFF_DenseDiagoSpaco,
  PointsSectionDFFF_DiagoDiagoDense,
  PointsSectionDFFF_DiagoDiagoDiago,
  PointsSectionDFFF_DiagoDiagoSpars,
  PointsSectionDFFF_DiagoDiagoSpaco,
  PointsSectionDFFF_SparsDiagoDense,
  PointsSectionDFFF_SparsDiagoDiago,
  PointsSectionDFFF_SparsDiagoSpars,
  PointsSectionDFFF_SparsDiagoSpaco,
  PointsSectionDFFF_SpacoDiagoDense,
  PointsSectionDFFF_SpacoDiagoDiago,
  PointsSectionDFFF_SpacoDiagoSpars,
  PointsSectionDFFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

