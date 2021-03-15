#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFFF : public Points2d3dBase < Points2d3dFFFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, numPoints2d, numPoints3d > {

public:


};

using PointsSectionFFFF_DenseDiagoDense = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_DenseDiagoDiago = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_DenseDiagoSpars = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_DenseDiagoSpaco = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DiagoDiagoDense = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_DiagoDiagoDiago = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_DiagoDiagoSpars = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_DiagoDiagoSpaco = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SparsDiagoDense = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_SparsDiagoDiago = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_SparsDiagoSpars = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_SparsDiagoSpaco = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SpacoDiagoDense = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_SpacoDiagoDiago = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_SpacoDiagoSpars = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_SpacoDiagoSpaco = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-FFFF", "[DoubleSection-FFFF]",
  PointsSectionFFFF_DenseDiagoDense,
  PointsSectionFFFF_DenseDiagoDiago,
  PointsSectionFFFF_DenseDiagoSpars,
  PointsSectionFFFF_DenseDiagoSpaco,
  PointsSectionFFFF_DiagoDiagoDense,
  PointsSectionFFFF_DiagoDiagoDiago,
  PointsSectionFFFF_DiagoDiagoSpars,
  PointsSectionFFFF_DiagoDiagoSpaco,
  PointsSectionFFFF_SparsDiagoDense,
  PointsSectionFFFF_SparsDiagoDiago,
  PointsSectionFFFF_SparsDiagoSpars,
  PointsSectionFFFF_SparsDiagoSpaco,
  PointsSectionFFFF_SpacoDiagoDense,
  PointsSectionFFFF_SpacoDiagoDiago,
  PointsSectionFFFF_SpacoDiagoSpars,
  PointsSectionFFFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

