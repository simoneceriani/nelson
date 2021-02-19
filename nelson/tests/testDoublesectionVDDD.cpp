#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionCommon.hpp"

#include "nelson/GaussNewton.hpp"
#include "nelson/LevenbergMarquardt.hpp"

#include <array>
#include <iostream>

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3d : public Points2d3dBase < Points2d3d<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

  std::vector<int> u_sizes;
public:

  Points2d3d() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }
  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }


  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSection_DenseDenseDense = Points2d3d<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSection_DenseDenseDiago = Points2d3d<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSection_DenseDenseSpars = Points2d3d<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSection_DenseDenseSpaco = Points2d3d<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSection_DenseDiagoDense = Points2d3d<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSection_DenseDiagoDiago = Points2d3d<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSection_DenseDiagoSpars = Points2d3d<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSection_DenseDiagoSpaco = Points2d3d<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSection_DenseSparsDense = Points2d3d<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSection_DenseSparsDiago = Points2d3d<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSection_DenseSparsSpars = Points2d3d<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSection_DenseSparsSpaco = Points2d3d<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSection_DenseSpacoDense = Points2d3d<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSection_DenseSpacoDiago = Points2d3d<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSection_DenseSpacoSpars = Points2d3d<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSection_DenseSpacoSpaco = Points2d3d<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSection_DiagoDenseDense = Points2d3d<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSection_DiagoDenseDiago = Points2d3d<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSection_DiagoDenseSpars = Points2d3d<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSection_DiagoDenseSpaco = Points2d3d<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSection_DiagoDiagoDense = Points2d3d<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSection_DiagoDiagoDiago = Points2d3d<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSection_DiagoDiagoSpars = Points2d3d<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSection_DiagoDiagoSpaco = Points2d3d<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSection_DiagoSparsDense = Points2d3d<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSection_DiagoSparsDiago = Points2d3d<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSection_DiagoSparsSpars = Points2d3d<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSection_DiagoSparsSpaco = Points2d3d<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSection_DiagoSpacoDense = Points2d3d<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSection_DiagoSpacoDiago = Points2d3d<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSection_DiagoSpacoSpars = Points2d3d<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSection_DiagoSpacoSpaco = Points2d3d<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSection_SparsDenseDense = Points2d3d<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSection_SparsDenseDiago = Points2d3d<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSection_SparsDenseSpars = Points2d3d<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSection_SparsDenseSpaco = Points2d3d<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSection_SparsDiagoDense = Points2d3d<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSection_SparsDiagoDiago = Points2d3d<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSection_SparsDiagoSpars = Points2d3d<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSection_SparsDiagoSpaco = Points2d3d<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSection_SparsSparsDense = Points2d3d<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSection_SparsSparsDiago = Points2d3d<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSection_SparsSparsSpars = Points2d3d<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSection_SparsSparsSpaco = Points2d3d<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSection_SparsSpacoDense = Points2d3d<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSection_SparsSpacoDiago = Points2d3d<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSection_SparsSpacoSpars = Points2d3d<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSection_SparsSpacoSpaco = Points2d3d<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSection_SpacoDenseDense = Points2d3d<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSection_SpacoDenseDiago = Points2d3d<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSection_SpacoDenseSpars = Points2d3d<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSection_SpacoDenseSpaco = Points2d3d<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSection_SpacoDiagoDense = Points2d3d<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSection_SpacoDiagoDiago = Points2d3d<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSection_SpacoDiagoSpars = Points2d3d<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSection_SpacoDiagoSpaco = Points2d3d<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSection_SpacoSparsDense = Points2d3d<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSection_SpacoSparsDiago = Points2d3d<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSection_SpacoSparsSpars = Points2d3d<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSection_SpacoSparsSpaco = Points2d3d<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSection_SpacoSpacoDense = Points2d3d<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSection_SpacoSpacoDiago = Points2d3d<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSection_SpacoSpacoSpars = Points2d3d<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSection_SpacoSpacoSpaco = Points2d3d<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-", "[DoubleSection-]",
  PointsSection_DenseDenseDense,
  PointsSection_DenseDenseDiago,
  PointsSection_DenseDenseSpars,
  PointsSection_DenseDenseSpaco,
  PointsSection_DenseDiagoDense,
  PointsSection_DenseDiagoDiago,
  PointsSection_DenseDiagoSpars,
  PointsSection_DenseDiagoSpaco,
  PointsSection_DenseSparsDense,
  PointsSection_DenseSparsDiago,
  PointsSection_DenseSparsSpars,
  PointsSection_DenseSparsSpaco,
  PointsSection_DenseSpacoDense,
  PointsSection_DenseSpacoDiago,
  PointsSection_DenseSpacoSpars,
  PointsSection_DenseSpacoSpaco,
  //------------------------------------
  PointsSection_DiagoDenseDense,
  PointsSection_DiagoDenseDiago,
  PointsSection_DiagoDenseSpars,
  PointsSection_DiagoDenseSpaco,
  PointsSection_DiagoDiagoDense,
  PointsSection_DiagoDiagoDiago,
  PointsSection_DiagoDiagoSpars,
  PointsSection_DiagoDiagoSpaco,
  PointsSection_DiagoSparsDense,
  PointsSection_DiagoSparsDiago,
  PointsSection_DiagoSparsSpars,
  PointsSection_DiagoSparsSpaco,
  PointsSection_DiagoSpacoDense,
  PointsSection_DiagoSpacoDiago,
  PointsSection_DiagoSpacoSpars,
  PointsSection_DiagoSpacoSpaco,
  //------------------------------------
  PointsSection_SparsDenseDense,
  PointsSection_SparsDenseDiago,
  PointsSection_SparsDenseSpars,
  PointsSection_SparsDenseSpaco,
  PointsSection_SparsDiagoDense,
  PointsSection_SparsDiagoDiago,
  PointsSection_SparsDiagoSpars,
  PointsSection_SparsDiagoSpaco,
  PointsSection_SparsSparsDense,
  PointsSection_SparsSparsDiago,
  PointsSection_SparsSparsSpars,
  PointsSection_SparsSparsSpaco,
  PointsSection_SparsSpacoDense,
  PointsSection_SparsSpacoDiago,
  PointsSection_SparsSpacoSpars,
  PointsSection_SparsSpacoSpaco,
  //------------------------------------
  PointsSection_SpacoDenseDense,
  PointsSection_SpacoDenseDiago,
  PointsSection_SpacoDenseSpars,
  PointsSection_SpacoDenseSpaco,
  PointsSection_SpacoDiagoDense,
  PointsSection_SpacoDiagoDiago,
  PointsSection_SpacoDiagoSpars,
  PointsSection_SpacoDiagoSpaco,
  PointsSection_SpacoSparsDense,
  PointsSection_SpacoSparsDiago,
  PointsSection_SpacoSparsSpars,
  PointsSection_SpacoSparsSpaco,
  PointsSection_SpacoSpacoDense,
  PointsSection_SpacoSpacoDiago,
  PointsSection_SpacoSpacoSpars,
  PointsSection_SpacoSpacoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}