#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/DoubleSectionHessian.hpp"
#include "nelson/DoubleSection.hpp"
#include "nelson/EdgeUnary.hpp"
#include "nelson/EdgeBinary.hpp"

#include <array>
#include <iostream>

struct Point2d {
  Eigen::Vector2d p2d;
  static constexpr int blockSize = 2;
};

struct Point3d {
  Eigen::Vector3d p3d;
  static constexpr int blockSize = 3;
};

static constexpr int numPoints2d = 5; // totsize = 10
static constexpr int numPoints3d = 3; // totsize = 9

template<int matTypeU, int matTypeV, int matTypeW>
class Points2d3dFDDF : public nelson::DoubleSection< Points2d3dFDDF<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, mat::Dynamic, mat::Dynamic, numPoints3d> {
  std::vector<Point2d> _points2d;
  std::array<Point3d, numPoints3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFDDF() {
    _points2d.resize(numPoints2d);
    this->parametersReady();
  }

  int parameterVSize() const override {
    return Point2d::blockSize;
  }
  int numParametersU() const override {
    return _points2d.size();
  }

  virtual const Point2d& parameterU(nelson::NodeId i) const override {
    if (i.isVariable()) return _points2d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint2d;
    }
  }
  virtual Point2d& parameterU(nelson::NodeId i) override {
    if (i.isVariable()) return _points2d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint2d;
    }
  }

  virtual const Point3d& parameterV(nelson::NodeId i) const override {
    if (i.isVariable()) return _points3d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint3d;
    }
  }
  virtual Point3d& parameterV(nelson::NodeId i) override {
    if (i.isVariable()) return _points3d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint3d;
    }
  }

  int numFixedParametersU() const override {
    return 1;
  }
  int numFixedParametersV() const override {
    return 1;
  }

};

using PointsSectionFDDF_DenseDenseDense = Points2d3dFDDF<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDF_DenseDenseDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseDenseSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDF_DenseDenseSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DenseDiagoDense = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DenseDiagoDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseDiagoSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DenseDiagoSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DenseSparsDense = Points2d3dFDDF<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDF_DenseSparsDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseSparsSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDF_DenseSparsSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DenseSpacoDense = Points2d3dFDDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDF_DenseSpacoDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseSpacoSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDF_DenseSpacoSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDF_DiagoDenseDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDF_DiagoDenseDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoDenseSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDF_DiagoDenseSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DiagoDiagoDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DiagoDiagoDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoDiagoSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DiagoDiagoSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DiagoSparsDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDF_DiagoSparsDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoSparsSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDF_DiagoSparsSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DiagoSpacoDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDF_DiagoSpacoDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoSpacoSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDF_DiagoSpacoSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDF_SparsDenseDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDF_SparsDenseDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsDenseSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDF_SparsDenseSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SparsDiagoDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SparsDiagoDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsDiagoSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SparsDiagoSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SparsSparsDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDF_SparsSparsDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsSparsSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDF_SparsSparsSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SparsSpacoDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDF_SparsSpacoDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsSpacoSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDF_SparsSpacoSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDF_SpacoDenseDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDF_SpacoDenseDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoDenseSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDF_SpacoDenseSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SpacoDiagoDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SpacoDiagoDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoDiagoSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SpacoDiagoSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SpacoSparsDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDF_SpacoSparsDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoSparsSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDF_SpacoSparsSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SpacoSpacoDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDF_SpacoSpacoDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoSpacoSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDF_SpacoSpacoSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDDF", "[DoubleSection-FDDF]",
  PointsSectionFDDF_DenseDenseDense,
  PointsSectionFDDF_DenseDenseDiago,
  PointsSectionFDDF_DenseDenseSpars,
  PointsSectionFDDF_DenseDenseSpaco,
  PointsSectionFDDF_DenseDiagoDense,
  PointsSectionFDDF_DenseDiagoDiago,
  PointsSectionFDDF_DenseDiagoSpars,
  PointsSectionFDDF_DenseDiagoSpaco,
  PointsSectionFDDF_DenseSparsDense,
  PointsSectionFDDF_DenseSparsDiago,
  PointsSectionFDDF_DenseSparsSpars,
  PointsSectionFDDF_DenseSparsSpaco,
  PointsSectionFDDF_DenseSpacoDense,
  PointsSectionFDDF_DenseSpacoDiago,
  PointsSectionFDDF_DenseSpacoSpars,
  PointsSectionFDDF_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFDDF_DiagoDenseDense,
  PointsSectionFDDF_DiagoDenseDiago,
  PointsSectionFDDF_DiagoDenseSpars,
  PointsSectionFDDF_DiagoDenseSpaco,
  PointsSectionFDDF_DiagoDiagoDense,
  PointsSectionFDDF_DiagoDiagoDiago,
  PointsSectionFDDF_DiagoDiagoSpars,
  PointsSectionFDDF_DiagoDiagoSpaco,
  PointsSectionFDDF_DiagoSparsDense,
  PointsSectionFDDF_DiagoSparsDiago,
  PointsSectionFDDF_DiagoSparsSpars,
  PointsSectionFDDF_DiagoSparsSpaco,
  PointsSectionFDDF_DiagoSpacoDense,
  PointsSectionFDDF_DiagoSpacoDiago,
  PointsSectionFDDF_DiagoSpacoSpars,
  PointsSectionFDDF_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFDDF_SparsDenseDense,
  PointsSectionFDDF_SparsDenseDiago,
  PointsSectionFDDF_SparsDenseSpars,
  PointsSectionFDDF_SparsDenseSpaco,
  PointsSectionFDDF_SparsDiagoDense,
  PointsSectionFDDF_SparsDiagoDiago,
  PointsSectionFDDF_SparsDiagoSpars,
  PointsSectionFDDF_SparsDiagoSpaco,
  PointsSectionFDDF_SparsSparsDense,
  PointsSectionFDDF_SparsSparsDiago,
  PointsSectionFDDF_SparsSparsSpars,
  PointsSectionFDDF_SparsSparsSpaco,
  PointsSectionFDDF_SparsSpacoDense,
  PointsSectionFDDF_SparsSpacoDiago,
  PointsSectionFDDF_SparsSpacoSpars,
  PointsSectionFDDF_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFDDF_SpacoDenseDense,
  PointsSectionFDDF_SpacoDenseDiago,
  PointsSectionFDDF_SpacoDenseSpars,
  PointsSectionFDDF_SpacoDenseSpaco,
  PointsSectionFDDF_SpacoDiagoDense,
  PointsSectionFDDF_SpacoDiagoDiago,
  PointsSectionFDDF_SpacoDiagoSpars,
  PointsSectionFDDF_SpacoDiagoSpaco,
  PointsSectionFDDF_SpacoSparsDense,
  PointsSectionFDDF_SpacoSparsDiago,
  PointsSectionFDDF_SpacoSparsSpars,
  PointsSectionFDDF_SpacoSparsSpaco,
  PointsSectionFDDF_SpacoSpacoDense,
  PointsSectionFDDF_SpacoSpacoDiago,
  PointsSectionFDDF_SpacoSpacoSpars,
  PointsSectionFDDF_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);

  pss.structureReady();

}