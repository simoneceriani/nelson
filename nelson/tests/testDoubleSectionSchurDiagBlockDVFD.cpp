#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVFD : public Points2d3dBase< Points2d3dDVFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, numPoints2d, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVFD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionDVFD_DenseDiagoDense = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_DenseDiagoDiago = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_DenseDiagoSpars = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_DenseDiagoSpaco = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_DiagoDiagoDense = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_DiagoDiagoDiago = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_DiagoDiagoSpars = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_DiagoDiagoSpaco = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_SparsDiagoDense = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_SparsDiagoDiago = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_SparsDiagoSpars = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_SparsDiagoSpaco = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_SpacoDiagoDense = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_SpacoDiagoDiago = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_SpacoDiagoSpars = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_SpacoDiagoSpaco = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DVFD", "[DoubleSection-DVFD]",
  PointsSectionDVFD_DenseDiagoDense,
  PointsSectionDVFD_DenseDiagoDiago,
  PointsSectionDVFD_DenseDiagoSpars,
  PointsSectionDVFD_DenseDiagoSpaco,
  PointsSectionDVFD_DiagoDiagoDense,
  PointsSectionDVFD_DiagoDiagoDiago,
  PointsSectionDVFD_DiagoDiagoSpars,
  PointsSectionDVFD_DiagoDiagoSpaco,
  PointsSectionDVFD_SparsDiagoDense,
  PointsSectionDVFD_SparsDiagoDiago,
  PointsSectionDVFD_SparsDiagoSpars,
  PointsSectionDVFD_SparsDiagoSpaco,
  PointsSectionDVFD_SpacoDiagoDense,
  PointsSectionDVFD_SpacoDiagoDiago,
  PointsSectionDVFD_SpacoDiagoSpars,
  PointsSectionDVFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

