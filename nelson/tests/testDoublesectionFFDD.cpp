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
class Points2d3dFFDD : public nelson::DoubleSection< Points2d3dFFDD<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, mat::Dynamic> {
  std::vector<Point2d> _points2d;
  std::vector<Point3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFFDD() {
    _points2d.resize(numPoints2d);
    _points3d.resize(numPoints3d);
    this->parametersReady();
  }

  int numParametersU() const override {
    return _points2d.size();
  }
  int numParametersV() const override {
    return _points3d.size();
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

using PointsSectionFFDD_DenseDenseDense = Points2d3dFFDD<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDD_DenseDenseDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseDenseSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDD_DenseDenseSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DenseDiagoDense = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DenseDiagoDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseDiagoSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DenseDiagoSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DenseSparsDense = Points2d3dFFDD<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDD_DenseSparsDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseSparsSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDD_DenseSparsSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DenseSpacoDense = Points2d3dFFDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDD_DenseSpacoDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseSpacoSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDD_DenseSpacoSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDD_DiagoDenseDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDD_DiagoDenseDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoDenseSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDD_DiagoDenseSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoDiagoDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DiagoDiagoDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoDiagoSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DiagoDiagoSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoSparsDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDD_DiagoSparsDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoSparsSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDD_DiagoSparsSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoSpacoDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDD_DiagoSpacoDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoSpacoSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDD_DiagoSpacoSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDD_SparsDenseDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDD_SparsDenseDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsDenseSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDD_SparsDenseSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsDiagoDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SparsDiagoDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsDiagoSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SparsDiagoSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsSparsDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDD_SparsSparsDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsSparsSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDD_SparsSparsSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsSpacoDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDD_SparsSpacoDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsSpacoSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDD_SparsSpacoSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFDD_SpacoDenseDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFDD_SpacoDenseDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoDenseSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFDD_SpacoDenseSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoDiagoDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SpacoDiagoDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoDiagoSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SpacoDiagoSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoSparsDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFDD_SpacoSparsDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoSparsSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFDD_SpacoSparsSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoSpacoDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFDD_SpacoSpacoDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoSpacoSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFDD_SpacoSpacoSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFDD", "[DoubleSection-FFDD]",
  PointsSectionFFDD_DenseDenseDense,
  PointsSectionFFDD_DenseDenseDiago,
  PointsSectionFFDD_DenseDenseSpars,
  PointsSectionFFDD_DenseDenseSpaco,
  PointsSectionFFDD_DenseDiagoDense,
  PointsSectionFFDD_DenseDiagoDiago,
  PointsSectionFFDD_DenseDiagoSpars,
  PointsSectionFFDD_DenseDiagoSpaco,
  PointsSectionFFDD_DenseSparsDense,
  PointsSectionFFDD_DenseSparsDiago,
  PointsSectionFFDD_DenseSparsSpars,
  PointsSectionFFDD_DenseSparsSpaco,
  PointsSectionFFDD_DenseSpacoDense,
  PointsSectionFFDD_DenseSpacoDiago,
  PointsSectionFFDD_DenseSpacoSpars,
  PointsSectionFFDD_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFFDD_DiagoDenseDense,
  PointsSectionFFDD_DiagoDenseDiago,
  PointsSectionFFDD_DiagoDenseSpars,
  PointsSectionFFDD_DiagoDenseSpaco,
  PointsSectionFFDD_DiagoDiagoDense,
  PointsSectionFFDD_DiagoDiagoDiago,
  PointsSectionFFDD_DiagoDiagoSpars,
  PointsSectionFFDD_DiagoDiagoSpaco,
  PointsSectionFFDD_DiagoSparsDense,
  PointsSectionFFDD_DiagoSparsDiago,
  PointsSectionFFDD_DiagoSparsSpars,
  PointsSectionFFDD_DiagoSparsSpaco,
  PointsSectionFFDD_DiagoSpacoDense,
  PointsSectionFFDD_DiagoSpacoDiago,
  PointsSectionFFDD_DiagoSpacoSpars,
  PointsSectionFFDD_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFFDD_SparsDenseDense,
  PointsSectionFFDD_SparsDenseDiago,
  PointsSectionFFDD_SparsDenseSpars,
  PointsSectionFFDD_SparsDenseSpaco,
  PointsSectionFFDD_SparsDiagoDense,
  PointsSectionFFDD_SparsDiagoDiago,
  PointsSectionFFDD_SparsDiagoSpars,
  PointsSectionFFDD_SparsDiagoSpaco,
  PointsSectionFFDD_SparsSparsDense,
  PointsSectionFFDD_SparsSparsDiago,
  PointsSectionFFDD_SparsSparsSpars,
  PointsSectionFFDD_SparsSparsSpaco,
  PointsSectionFFDD_SparsSpacoDense,
  PointsSectionFFDD_SparsSpacoDiago,
  PointsSectionFFDD_SparsSpacoSpars,
  PointsSectionFFDD_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFFDD_SpacoDenseDense,
  PointsSectionFFDD_SpacoDenseDiago,
  PointsSectionFFDD_SpacoDenseSpars,
  PointsSectionFFDD_SpacoDenseSpaco,
  PointsSectionFFDD_SpacoDiagoDense,
  PointsSectionFFDD_SpacoDiagoDiago,
  PointsSectionFFDD_SpacoDiagoSpars,
  PointsSectionFFDD_SpacoDiagoSpaco,
  PointsSectionFFDD_SpacoSparsDense,
  PointsSectionFFDD_SpacoSparsDiago,
  PointsSectionFFDD_SpacoSparsSpars,
  PointsSectionFFDD_SpacoSparsSpaco,
  PointsSectionFFDD_SpacoSpacoDense,
  PointsSectionFFDD_SpacoSpacoDiago,
  PointsSectionFFDD_SpacoSpacoSpars,
  PointsSectionFFDD_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);



}