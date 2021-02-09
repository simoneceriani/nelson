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
class Points2d3dFFDF : public nelson::DoubleSection< Points2d3dFFDF<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, numPoints3d> {
  std::vector<Point2d> _points2d;
  std::array<Point3d, numPoints3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFFDF() {
    _points2d.resize(numPoints2d);
    this->parametersReady();
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

using PointsSectionFFDF_DenseDenseDense = Points2d3dFFDF<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDF_DenseDenseDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseDenseSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDF_DenseDenseSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DenseDiagoDense = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DenseDiagoDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseDiagoSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DenseDiagoSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DenseSparsDense = Points2d3dFFDF<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDF_DenseSparsDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseSparsSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDF_DenseSparsSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DenseSpacoDense = Points2d3dFFDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDF_DenseSpacoDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseSpacoSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDF_DenseSpacoSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDF_DiagoDenseDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDF_DiagoDenseDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoDenseSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDF_DiagoDenseSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DiagoDiagoDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DiagoDiagoDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoDiagoSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DiagoDiagoSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DiagoSparsDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDF_DiagoSparsDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoSparsSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDF_DiagoSparsSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DiagoSpacoDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDF_DiagoSpacoDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoSpacoSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDF_DiagoSpacoSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDF_SparsDenseDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDF_SparsDenseDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsDenseSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDF_SparsDenseSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SparsDiagoDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SparsDiagoDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsDiagoSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SparsDiagoSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SparsSparsDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDF_SparsSparsDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsSparsSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDF_SparsSparsSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SparsSpacoDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDF_SparsSpacoDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsSpacoSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDF_SparsSpacoSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDF_SpacoDenseDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDF_SpacoDenseDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoDenseSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDF_SpacoDenseSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SpacoDiagoDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SpacoDiagoDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoDiagoSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SpacoDiagoSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SpacoSparsDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDF_SpacoSparsDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoSparsSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDF_SpacoSparsSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SpacoSpacoDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDF_SpacoSpacoDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoSpacoSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDF_SpacoSpacoSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFDF", "[DoubleSection-FFDF]",
  PointsSectionFFDF_DenseDenseDense,
  PointsSectionFFDF_DenseDenseDiago,
  PointsSectionFFDF_DenseDenseSpars,
  PointsSectionFFDF_DenseDenseSpaco,
  PointsSectionFFDF_DenseDiagoDense,
  PointsSectionFFDF_DenseDiagoDiago,
  PointsSectionFFDF_DenseDiagoSpars,
  PointsSectionFFDF_DenseDiagoSpaco,
  PointsSectionFFDF_DenseSparsDense,
  PointsSectionFFDF_DenseSparsDiago,
  PointsSectionFFDF_DenseSparsSpars,
  PointsSectionFFDF_DenseSparsSpaco,
  PointsSectionFFDF_DenseSpacoDense,
  PointsSectionFFDF_DenseSpacoDiago,
  PointsSectionFFDF_DenseSpacoSpars,
  PointsSectionFFDF_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFFDF_DiagoDenseDense,
  PointsSectionFFDF_DiagoDenseDiago,
  PointsSectionFFDF_DiagoDenseSpars,
  PointsSectionFFDF_DiagoDenseSpaco,
  PointsSectionFFDF_DiagoDiagoDense,
  PointsSectionFFDF_DiagoDiagoDiago,
  PointsSectionFFDF_DiagoDiagoSpars,
  PointsSectionFFDF_DiagoDiagoSpaco,
  PointsSectionFFDF_DiagoSparsDense,
  PointsSectionFFDF_DiagoSparsDiago,
  PointsSectionFFDF_DiagoSparsSpars,
  PointsSectionFFDF_DiagoSparsSpaco,
  PointsSectionFFDF_DiagoSpacoDense,
  PointsSectionFFDF_DiagoSpacoDiago,
  PointsSectionFFDF_DiagoSpacoSpars,
  PointsSectionFFDF_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFFDF_SparsDenseDense,
  PointsSectionFFDF_SparsDenseDiago,
  PointsSectionFFDF_SparsDenseSpars,
  PointsSectionFFDF_SparsDenseSpaco,
  PointsSectionFFDF_SparsDiagoDense,
  PointsSectionFFDF_SparsDiagoDiago,
  PointsSectionFFDF_SparsDiagoSpars,
  PointsSectionFFDF_SparsDiagoSpaco,
  PointsSectionFFDF_SparsSparsDense,
  PointsSectionFFDF_SparsSparsDiago,
  PointsSectionFFDF_SparsSparsSpars,
  PointsSectionFFDF_SparsSparsSpaco,
  PointsSectionFFDF_SparsSpacoDense,
  PointsSectionFFDF_SparsSpacoDiago,
  PointsSectionFFDF_SparsSpacoSpars,
  PointsSectionFFDF_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFFDF_SpacoDenseDense,
  PointsSectionFFDF_SpacoDenseDiago,
  PointsSectionFFDF_SpacoDenseSpars,
  PointsSectionFFDF_SpacoDenseSpaco,
  PointsSectionFFDF_SpacoDiagoDense,
  PointsSectionFFDF_SpacoDiagoDiago,
  PointsSectionFFDF_SpacoDiagoSpars,
  PointsSectionFFDF_SpacoDiagoSpaco,
  PointsSectionFFDF_SpacoSparsDense,
  PointsSectionFFDF_SpacoSparsDiago,
  PointsSectionFFDF_SpacoSparsSpars,
  PointsSectionFFDF_SpacoSparsSpaco,
  PointsSectionFFDF_SpacoSpacoDense,
  PointsSectionFFDF_SpacoSpacoDiago,
  PointsSectionFFDF_SpacoSpacoSpars,
  PointsSectionFFDF_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);



}