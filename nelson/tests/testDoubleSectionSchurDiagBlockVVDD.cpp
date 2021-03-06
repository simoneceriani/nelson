#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVVDD : public Points2d3dBase < Points2d3dVVDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic > {
  std::vector<int> v_sizes;
  std::vector<int> u_sizes;
public:

  Points2d3dVVDD() : v_sizes(numPoints3d, Point3d::blockSize), u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionVVDD_DenseDiagoDense = Points2d3dVVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDD_DenseDiagoDiago = Points2d3dVVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDD_DenseDiagoSpars = Points2d3dVVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDD_DenseDiagoSpaco = Points2d3dVVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDD_DiagoDiagoDense = Points2d3dVVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDD_DiagoDiagoDiago = Points2d3dVVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDD_DiagoDiagoSpars = Points2d3dVVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDD_DiagoDiagoSpaco = Points2d3dVVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDD_SparsDiagoDense = Points2d3dVVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDD_SparsDiagoDiago = Points2d3dVVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDD_SparsDiagoSpars = Points2d3dVVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDD_SparsDiagoSpaco = Points2d3dVVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVDD_SpacoDiagoDense = Points2d3dVVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVDD_SpacoDiagoDiago = Points2d3dVVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVDD_SpacoDiagoSpars = Points2d3dVVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVDD_SpacoDiagoSpaco = Points2d3dVVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VVDD", "[DoubleSection-VVDD]",
  PointsSectionVVDD_DenseDiagoDense,
  PointsSectionVVDD_DenseDiagoDiago,
  PointsSectionVVDD_DenseDiagoSpars,
  PointsSectionVVDD_DenseDiagoSpaco,
  PointsSectionVVDD_DiagoDiagoDense,
  PointsSectionVVDD_DiagoDiagoDiago,
  PointsSectionVVDD_DiagoDiagoSpars,
  PointsSectionVVDD_DiagoDiagoSpaco,
  PointsSectionVVDD_SparsDiagoDense,
  PointsSectionVVDD_SparsDiagoDiago,
  PointsSectionVVDD_SparsDiagoSpars,
  PointsSectionVVDD_SparsDiagoSpaco,
  PointsSectionVVDD_SpacoDiagoDense,
  PointsSectionVVDD_SpacoDiagoDiago,
  PointsSectionVVDD_SpacoDiagoSpars,
  PointsSectionVVDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

