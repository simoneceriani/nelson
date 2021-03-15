#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVVFD : public Points2d3dBase < Points2d3dVVFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Variable, numPoints2d, mat::Dynamic > {
  std::vector<int> v_sizes;
  std::vector<int> u_sizes;
public:

  Points2d3dVVFD() : v_sizes(numPoints3d, Point3d::blockSize), u_sizes(numPoints2d, Point2d::blockSize) { }

  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionVVFD_DenseDiagoDense = Points2d3dVVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFD_DenseDiagoDiago = Points2d3dVVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFD_DenseDiagoSpars = Points2d3dVVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFD_DenseDiagoSpaco = Points2d3dVVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFD_DiagoDiagoDense = Points2d3dVVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFD_DiagoDiagoDiago = Points2d3dVVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFD_DiagoDiagoSpars = Points2d3dVVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFD_DiagoDiagoSpaco = Points2d3dVVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFD_SparsDiagoDense = Points2d3dVVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFD_SparsDiagoDiago = Points2d3dVVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFD_SparsDiagoSpars = Points2d3dVVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFD_SparsDiagoSpaco = Points2d3dVVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFD_SpacoDiagoDense = Points2d3dVVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFD_SpacoDiagoDiago = Points2d3dVVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFD_SpacoDiagoSpars = Points2d3dVVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFD_SpacoDiagoSpaco = Points2d3dVVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VVFD", "[DoubleSection-VVFD]",
  PointsSectionVVFD_DenseDiagoDense,
  PointsSectionVVFD_DenseDiagoDiago,
  PointsSectionVVFD_DenseDiagoSpars,
  PointsSectionVVFD_DenseDiagoSpaco,
  PointsSectionVVFD_DiagoDiagoDense,
  PointsSectionVVFD_DiagoDiagoDiago,
  PointsSectionVVFD_DiagoDiagoSpars,
  PointsSectionVVFD_DiagoDiagoSpaco,
  PointsSectionVVFD_SparsDiagoDense,
  PointsSectionVVFD_SparsDiagoDiago,
  PointsSectionVVFD_SparsDiagoSpars,
  PointsSectionVVFD_SparsDiagoSpaco,
  PointsSectionVVFD_SpacoDiagoDense,
  PointsSectionVVFD_SpacoDiagoDiago,
  PointsSectionVVFD_SpacoDiagoSpars,
  PointsSectionVVFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

