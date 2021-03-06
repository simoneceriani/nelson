#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVDF : public Points2d3dBase < Points2d3dDVDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, mat::Dynamic, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVDF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDVDF_DenseDiagoDense = Points2d3dDVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDF_DenseDiagoDiago = Points2d3dDVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDF_DenseDiagoSpars = Points2d3dDVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDF_DenseDiagoSpaco = Points2d3dDVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDF_DiagoDiagoDense = Points2d3dDVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDF_DiagoDiagoDiago = Points2d3dDVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDF_DiagoDiagoSpars = Points2d3dDVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDF_DiagoDiagoSpaco = Points2d3dDVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDF_SparsDiagoDense = Points2d3dDVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDF_SparsDiagoDiago = Points2d3dDVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDF_SparsDiagoSpars = Points2d3dDVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDF_SparsDiagoSpaco = Points2d3dDVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDF_SpacoDiagoDense = Points2d3dDVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDF_SpacoDiagoDiago = Points2d3dDVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDF_SpacoDiagoSpars = Points2d3dDVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDF_SpacoDiagoSpaco = Points2d3dDVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DVDF", "[DoubleSection-DVDF]",
  PointsSectionDVDF_DenseDiagoDense,
  PointsSectionDVDF_DenseDiagoDiago,
  PointsSectionDVDF_DenseDiagoSpars,
  PointsSectionDVDF_DenseDiagoSpaco,
  PointsSectionDVDF_DiagoDiagoDense,
  PointsSectionDVDF_DiagoDiagoDiago,
  PointsSectionDVDF_DiagoDiagoSpars,
  PointsSectionDVDF_DiagoDiagoSpaco,
  PointsSectionDVDF_SparsDiagoDense,
  PointsSectionDVDF_SparsDiagoDiago,
  PointsSectionDVDF_SparsDiagoSpars,
  PointsSectionDVDF_SparsDiagoSpaco,
  PointsSectionDVDF_SpacoDiagoDense,
  PointsSectionDVDF_SpacoDiagoDiago,
  PointsSectionDVDF_SpacoDiagoSpars,
  PointsSectionDVDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

