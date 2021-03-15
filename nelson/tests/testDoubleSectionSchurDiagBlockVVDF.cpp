#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVVDF : public Points2d3dBase < Points2d3dVVDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Variable, mat::Dynamic, numPoints3d > {
  std::vector<int> v_sizes;
  std::vector<int> u_sizes;
public:

  Points2d3dVVDF() : v_sizes(numPoints3d, Point3d::blockSize), u_sizes(numPoints2d, Point2d::blockSize) { }

  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionVVDF_DenseDiagoDense = Points2d3dVVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDF_DenseDiagoDiago = Points2d3dVVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDF_DenseDiagoSpars = Points2d3dVVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDF_DenseDiagoSpaco = Points2d3dVVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDF_DiagoDiagoDense = Points2d3dVVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDF_DiagoDiagoDiago = Points2d3dVVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDF_DiagoDiagoSpars = Points2d3dVVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDF_DiagoDiagoSpaco = Points2d3dVVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDF_SparsDiagoDense = Points2d3dVVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDF_SparsDiagoDiago = Points2d3dVVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDF_SparsDiagoSpars = Points2d3dVVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDF_SparsDiagoSpaco = Points2d3dVVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDF_SpacoDiagoDense = Points2d3dVVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDF_SpacoDiagoDiago = Points2d3dVVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDF_SpacoDiagoSpars = Points2d3dVVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDF_SpacoDiagoSpaco = Points2d3dVVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VVDF", "[DoubleSection-VVDF]",
  PointsSectionVVDF_DenseDiagoDense,
  PointsSectionVVDF_DenseDiagoDiago,
  PointsSectionVVDF_DenseDiagoSpars,
  PointsSectionVVDF_DenseDiagoSpaco,
  PointsSectionVVDF_DiagoDiagoDense,
  PointsSectionVVDF_DiagoDiagoDiago,
  PointsSectionVVDF_DiagoDiagoSpars,
  PointsSectionVVDF_DiagoDiagoSpaco,
  PointsSectionVVDF_SparsDiagoDense,
  PointsSectionVVDF_SparsDiagoDiago,
  PointsSectionVVDF_SparsDiagoSpars,
  PointsSectionVVDF_SparsDiagoSpaco,
  PointsSectionVVDF_SpacoDiagoDense,
  PointsSectionVVDF_SpacoDiagoDiago,
  PointsSectionVVDF_SpacoDiagoSpars,
  PointsSectionVVDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

