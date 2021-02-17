#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/DoubleSectionHessian.hpp"
#include "nelson/DoubleSection.hpp"
#include "nelson/EdgeUnary.hpp"
#include "nelson/EdgeBinary.hpp"

#include "nelson/GaussNewton.hpp"
#include "nelson/LevenbergMarquardt.hpp"

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


template<class Section>
class EdgeUnaryPoint2d : public nelson::EdgeUnarySectionBaseCRPT<Section, typename Section::EdgeUnaryUAdapter, EdgeUnaryPoint2d<Section>> {
  int _parId;
  Eigen::Vector2d _meas_p2d;

  Eigen::Vector2d _err;
public:
  EdgeUnaryPoint2d(
    int parId,
    const Eigen::Vector2d& meas_p2d
  ) : _parId(parId),
    _meas_p2d(meas_p2d)
  {

  }


  void update(bool hessians) override {
    if (this->parId().isVariable()) {
      REQUIRE(this->parId().id() == _parId);
      REQUIRE(this->HUid() >= 0);
    }
    else {
      REQUIRE(this->parId().id() == _parId);
    }

    const auto& par = this->parameter();
    _err = par.p2d - _meas_p2d;

    this->setChi2(_err.squaredNorm());
  }

  template<class Derived1, class Derived2>
  void updateHBlock(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    REQUIRE(this->parId().isVariable());
    // std::cout << "EdgeUnaryTest::updateHBlock " << this->parId().id() << "," << this->parId().id() << std::endl;
    H += Eigen::Matrix2d::Identity();
    b += _err;
  }  

};

template<class Section>
class EdgeBinaryPoint2d : public nelson::EdgeBinarySectionBaseCRPT<Section, typename Section::EdgeBinaryAdapterUU, EdgeBinaryPoint2d<Section>> {
  int _par1Id;
  int _par2Id;

  Eigen::Vector2d _meas_distPoints2d;

  Eigen::Vector2d _err;
public:
  EdgeBinaryPoint2d(
    int par1Id,
    int par2Id,
    const Eigen::Vector2d& meas_distPoints2d
  ) : _par1Id(par1Id),
    _par2Id(par2Id),
    _meas_distPoints2d(meas_distPoints2d)
  {

  }


  void update(bool hessians) override {
    if (this->par_1_Id().isVariable()) {
      REQUIRE(this->par_1_Id().id() == _par1Id);
      REQUIRE(this->H_11_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_1_Id().id() == _par1Id);
    }
    if (this->par_2_Id().isVariable()) {
      REQUIRE(this->par_2_Id().id() == _par2Id);
      REQUIRE(this->H_22_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_2_Id().id() == _par2Id);
    }
    if (this->par_1_Id().isVariable() && this->par_2_Id().isVariable()) {
      REQUIRE(this->H_12_Uid() >= 0);
    }

    const auto& par1 = this->parameter_1();
    const auto& par2 = this->parameter_2();
    
    _err = (par1.p2d - par2.p2d) - _meas_distPoints2d;
    this->setChi2(_err.squaredNorm());
  }

  template<class Derived1, class Derived2>
  void updateH11Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH11Block " << this->par_1_Id().id() << "," << this->par_1_Id().id() << std::endl;
    H += Eigen::Matrix2d::Identity();
    v += _err;
  }
  template<class Derived1, class Derived2>
  void updateH22Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH22Block " << this->par_2_Id().id() << "," << this->par_2_Id().id() << std::endl;
    H += Eigen::Matrix2d::Identity();
    v -= _err;
  }
  template<class Derived1>
  void updateH12Block(Eigen::MatrixBase<Derived1>& H) {
    // std::cout << "EdgeBinartTest::updateH12Block " << this->par_1_Id().id() << "," << this->par_2_Id().id() << std::endl;
    H -= Eigen::Matrix2d::Identity();
  }


};


template<class Section>
class EdgeUnaryPoint3d : public nelson::EdgeUnarySectionBaseCRPT<Section, typename Section::EdgeUnaryVAdapter, EdgeUnaryPoint3d<Section>> {
  int _parId;
  Eigen::Vector3d _meas_p3d;

  Eigen::Vector3d _err;
public:
  EdgeUnaryPoint3d(
    int parId,
    const Eigen::Vector3d& meas_p3d
  ) : _parId(parId),
    _meas_p3d(meas_p3d)
  {

  }


  void update(bool hessians) override {
    if (this->parId().isVariable()) {
      REQUIRE(this->parId().id() == _parId);
      REQUIRE(this->HUid() >= 0);
    }
    else {
      REQUIRE(this->parId().id() == _parId);
    }

    const auto& par = this->parameter();
    _err = par.p3d - _meas_p3d;
    this->setChi2(_err.squaredNorm());
  }

  template<class Derived1, class Derived2>
  void updateHBlock(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    REQUIRE(this->parId().isVariable());
    // std::cout << "EdgeUnaryTest::updateHBlock " << this->parId().id() << "," << this->parId().id() << std::endl;
    H += Eigen::Matrix3d::Identity();
    v += _err;
  }


};

template<class Section>
class EdgeBinaryPoint3d : public nelson::EdgeBinarySectionBaseCRPT<Section, typename Section::EdgeBinaryAdapterVV, EdgeBinaryPoint3d<Section>> {
  int _par1Id;
  int _par2Id;

  Eigen::Vector3d _meas_distPoints3d;

  Eigen::Vector3d _err;
public:
  EdgeBinaryPoint3d(
    int par1Id,
    int par2Id,
    const Eigen::Vector3d& meas_distPoints3d
  ) : _par1Id(par1Id),
    _par2Id(par2Id),
    _meas_distPoints3d(meas_distPoints3d)
  {

  }


  void update(bool hessians) override {
    if (this->par_1_Id().isVariable()) {
      REQUIRE(this->par_1_Id().id() == _par1Id);
      REQUIRE(this->H_11_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_1_Id().id() == _par1Id);
    }
    if (this->par_2_Id().isVariable()) {
      REQUIRE(this->par_2_Id().id() == _par2Id);
      REQUIRE(this->H_22_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_2_Id().id() == _par2Id);
    }
    if (this->par_1_Id().isVariable() && this->par_2_Id().isVariable()) {
      REQUIRE(this->H_12_Uid() >= 0);
    }

    const auto& par1 = this->parameter_1();
    const auto& par2 = this->parameter_2();

    _err = (par1.p3d - par2.p3d) - _meas_distPoints3d;
    this->setChi2(_err.squaredNorm());
  }

  template<class Derived1, class Derived2>
  void updateH11Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH11Block " << this->par_1_Id().id() << "," << this->par_1_Id().id() << std::endl;
    H += Eigen::Matrix3d::Identity();
    v += _err;
  }
  template<class Derived1, class Derived2>
  void updateH22Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH22Block " << this->par_2_Id().id() << "," << this->par_2_Id().id() << std::endl;
    H += Eigen::Matrix3d::Identity();
    v -= _err;
  }
  template<class Derived1>
  void updateH12Block(Eigen::MatrixBase<Derived1>& H) {
    // std::cout << "EdgeBinartTest::updateH12Block " << this->par_1_Id().id() << "," << this->par_2_Id().id() << std::endl;
    H -= Eigen::Matrix3d::Identity();
  }


};

template<class Section>
class EdgeBinaryPoint2d3d : public nelson::EdgeBinarySectionBaseCRPT<Section, typename Section::EdgeBinaryAdapterUV, EdgeBinaryPoint2d3d<Section>> {
  int _par1Id;
  int _par2Id;

  Eigen::Vector2d _meas_distPoints2d;

  Eigen::Vector2d _err;
public:
  EdgeBinaryPoint2d3d(
    int par1Id,
    int par2Id,
    const Eigen::Vector2d& meas_distPoints2d
  ) : _par1Id(par1Id),
    _par2Id(par2Id),
    _meas_distPoints2d(meas_distPoints2d)
  {

  }


  void update(bool hessians) override {
    if (this->par_1_Id().isVariable()) {
      REQUIRE(this->par_1_Id().id() == _par1Id);
      REQUIRE(this->H_11_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_1_Id().id() == _par1Id);
    }
    if (this->par_2_Id().isVariable()) {
      REQUIRE(this->par_2_Id().id() == _par2Id);
      REQUIRE(this->H_22_Uid() >= 0);
    }
    else {
      REQUIRE(this->par_2_Id().id() == _par2Id);
    }
    if (this->par_1_Id().isVariable() && this->par_2_Id().isVariable()) {
      REQUIRE(this->H_12_Uid() >= 0);
    }

    const auto& par1 = this->parameter_1();
    const auto& par2 = this->parameter_2();

    _err = (par1.p2d - par2.p3d.template head<2>()) - _meas_distPoints2d;
    this->setChi2(_err.squaredNorm());
  }

  template<class Derived1, class Derived2>
  void updateH11Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH11Block " << this->par_1_Id().id() << "," << this->par_1_Id().id() << std::endl;
    H += Eigen::Matrix2d::Identity();
    v += _err;
  }
  template<class Derived1, class Derived2>
  void updateH22Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& v) {
    // std::cout << "EdgeBinartTest::updateH22Block " << this->par_2_Id().id() << "," << this->par_2_Id().id() << std::endl;

    H += Eigen::Vector3d(1,1,0).asDiagonal();
    v -= Eigen::Vector3d(_err.x(), _err.y(), 0);
  }
  template<class Derived1>
  void updateH12Block(Eigen::MatrixBase<Derived1>& H) {
    // std::cout << "EdgeBinartTest::updateH12Block " << this->par_1_Id().id() << "," << this->par_2_Id().id() << std::endl;

    Eigen::Matrix<double, 2, 3> Htmp;
    Htmp.setZero();
    Htmp(0, 0) = -1;
    Htmp(1, 1) = -1;
    H += Htmp;
  }


};


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFFF : public nelson::DoubleSection< Points2d3dFFFF<matTypeUv, matTypeVv, matTypeWv>, Point2d, Point3d, matTypeUv, matTypeVv, matTypeWv, double, Point2d::blockSize, Point3d::blockSize, numPoints2d, numPoints3d> {

  std::array<Point2d, numPoints2d> _points2d;
  std::array<Point3d, numPoints3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;

  using DoubleSectionBase = nelson::DoubleSection< Points2d3dFFFF<matTypeUv, matTypeVv, matTypeWv>, Point2d, Point3d, matTypeUv, matTypeVv, matTypeWv, double, Point2d::blockSize, Point3d::blockSize, numPoints2d, numPoints3d>;

public:
  Points2d3dFFFF() {
    Eigen::Matrix3Xd groundThruthPoints;
    groundThruthPoints.setRandom(3, std::max(numPoints2d, numPoints3d));

    for (int i = 0; i < numPoints2d; i++) { _points2d[i].p2d = groundThruthPoints.col(i).template head<2>(); }
    for (int i = 0; i < numPoints3d; i++) { _points3d[i].p3d = groundThruthPoints.col(i); }
    _fixedPoint2d.p2d.setRandom();
    _fixedPoint3d.p3d.setRandom();
    this->parametersReady();
  }

  void addNoise(double sigma) {
    for (int i = 0; i < numPoints2d; i++) {
      _points2d[i].p2d += Eigen::Vector2d::Random() * sigma;
    }
    for (int i = 0; i < numPoints3d; i++) {
      _points3d[i].p3d += Eigen::Vector3d::Random() * sigma;
    }
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

  void oplus(const typename DoubleSectionBase::HessianVectorsT& inc) {
    for (int i = 0; i < numPoints2d; i++) {
      _points2d[i].p2d += inc.bU().segment(i);
    }
    for (int i = 0; i < numPoints3d; i++) {
      _points3d[i].p3d += inc.bV().segment(i);
    }
  }


};

using PointsSectionFFFF_DenseDenseDense = Points2d3dFFFF<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFF_DenseDenseDiago = Points2d3dFFFF<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFF_DenseDenseSpars = Points2d3dFFFF<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFF_DenseDenseSpaco = Points2d3dFFFF<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DenseDiagoDense = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_DenseDiagoDiago = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_DenseDiagoSpars = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_DenseDiagoSpaco = Points2d3dFFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DenseSparsDense = Points2d3dFFFF<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFF_DenseSparsDiago = Points2d3dFFFF<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_DenseSparsSpars = Points2d3dFFFF<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFF_DenseSparsSpaco = Points2d3dFFFF<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DenseSpacoDense = Points2d3dFFFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFF_DenseSpacoDiago = Points2d3dFFFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_DenseSpacoSpars = Points2d3dFFFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFF_DenseSpacoSpaco = Points2d3dFFFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFF_DiagoDenseDense = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFF_DiagoDenseDiago = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFF_DiagoDenseSpars = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFF_DiagoDenseSpaco = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DiagoDiagoDense = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_DiagoDiagoDiago = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_DiagoDiagoSpars = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_DiagoDiagoSpaco = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DiagoSparsDense = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFF_DiagoSparsDiago = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_DiagoSparsSpars = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFF_DiagoSparsSpaco = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFF_DiagoSpacoDense = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFF_DiagoSpacoDiago = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_DiagoSpacoSpars = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFF_DiagoSpacoSpaco = Points2d3dFFFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFF_SparsDenseDense = Points2d3dFFFF<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFF_SparsDenseDiago = Points2d3dFFFF<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFF_SparsDenseSpars = Points2d3dFFFF<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFF_SparsDenseSpaco = Points2d3dFFFF<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SparsDiagoDense = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_SparsDiagoDiago = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_SparsDiagoSpars = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_SparsDiagoSpaco = Points2d3dFFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SparsSparsDense = Points2d3dFFFF<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFF_SparsSparsDiago = Points2d3dFFFF<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_SparsSparsSpars = Points2d3dFFFF<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFF_SparsSparsSpaco = Points2d3dFFFF<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SparsSpacoDense = Points2d3dFFFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFF_SparsSpacoDiago = Points2d3dFFFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_SparsSpacoSpars = Points2d3dFFFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFF_SparsSpacoSpaco = Points2d3dFFFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFF_SpacoDenseDense = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFF_SpacoDenseDiago = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFF_SpacoDenseSpars = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFF_SpacoDenseSpaco = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SpacoDiagoDense = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFF_SpacoDiagoDiago = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFF_SpacoDiagoSpars = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFF_SpacoDiagoSpaco = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SpacoSparsDense = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFF_SpacoSparsDiago = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_SpacoSparsSpars = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFF_SpacoSparsSpaco = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFF_SpacoSpacoDense = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFF_SpacoSpacoDiago = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFF_SpacoSpacoSpars = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFF_SpacoSpacoSpaco = Points2d3dFFFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFFF", "[DoubleSection-FFFF]",
  PointsSectionFFFF_DenseDenseDense,
  PointsSectionFFFF_DenseDenseDiago,
  PointsSectionFFFF_DenseDenseSpars,
  PointsSectionFFFF_DenseDenseSpaco,
  PointsSectionFFFF_DenseDiagoDense,
  PointsSectionFFFF_DenseDiagoDiago,
  PointsSectionFFFF_DenseDiagoSpars,
  PointsSectionFFFF_DenseDiagoSpaco,
  PointsSectionFFFF_DenseSparsDense,
  PointsSectionFFFF_DenseSparsDiago,
  PointsSectionFFFF_DenseSparsSpars,
  PointsSectionFFFF_DenseSparsSpaco,
  PointsSectionFFFF_DenseSpacoDense,
  PointsSectionFFFF_DenseSpacoDiago,
  PointsSectionFFFF_DenseSpacoSpars,
  PointsSectionFFFF_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFFFF_DiagoDenseDense,
  PointsSectionFFFF_DiagoDenseDiago,
  PointsSectionFFFF_DiagoDenseSpars,
  PointsSectionFFFF_DiagoDenseSpaco,
  PointsSectionFFFF_DiagoDiagoDense,
  PointsSectionFFFF_DiagoDiagoDiago,
  PointsSectionFFFF_DiagoDiagoSpars,
  PointsSectionFFFF_DiagoDiagoSpaco,
  PointsSectionFFFF_DiagoSparsDense,
  PointsSectionFFFF_DiagoSparsDiago,
  PointsSectionFFFF_DiagoSparsSpars,
  PointsSectionFFFF_DiagoSparsSpaco,
  PointsSectionFFFF_DiagoSpacoDense,
  PointsSectionFFFF_DiagoSpacoDiago,
  PointsSectionFFFF_DiagoSpacoSpars,
  PointsSectionFFFF_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFFFF_SparsDenseDense,
  PointsSectionFFFF_SparsDenseDiago,
  PointsSectionFFFF_SparsDenseSpars,
  PointsSectionFFFF_SparsDenseSpaco,
  PointsSectionFFFF_SparsDiagoDense,
  PointsSectionFFFF_SparsDiagoDiago,
  PointsSectionFFFF_SparsDiagoSpars,
  PointsSectionFFFF_SparsDiagoSpaco,
  PointsSectionFFFF_SparsSparsDense,
  PointsSectionFFFF_SparsSparsDiago,
  PointsSectionFFFF_SparsSparsSpars,
  PointsSectionFFFF_SparsSparsSpaco,
  PointsSectionFFFF_SparsSpacoDense,
  PointsSectionFFFF_SparsSpacoDiago,
  PointsSectionFFFF_SparsSpacoSpars,
  PointsSectionFFFF_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFFFF_SpacoDenseDense,
  PointsSectionFFFF_SpacoDenseDiago,
  PointsSectionFFFF_SpacoDenseSpars,
  PointsSectionFFFF_SpacoDenseSpaco,
  PointsSectionFFFF_SpacoDiagoDense,
  PointsSectionFFFF_SpacoDiagoDiago,
  PointsSectionFFFF_SpacoDiagoSpars,
  PointsSectionFFFF_SpacoDiagoSpaco,
  PointsSectionFFFF_SpacoSparsDense,
  PointsSectionFFFF_SpacoSparsDiago,
  PointsSectionFFFF_SpacoSparsSpars,
  PointsSectionFFFF_SpacoSparsSpaco,
  PointsSectionFFFF_SpacoSpacoDense,
  PointsSectionFFFF_SpacoSpacoDiago,
  PointsSectionFFFF_SpacoSpacoSpars,
  PointsSectionFFFF_SpacoSpacoSpaco 
)
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);

  // unary edge first section
  for (int i = 0; i < numPoints2d; i++) {
    pss.addEdge(i, new EdgeUnaryPoint2d<TestType>(i, pss.parameterU(i).p2d));
  }
  // unary edge second section
  for (int i = 0; i < numPoints3d; i++) {
    pss.addEdge(i, new EdgeUnaryPoint3d<TestType>(i, pss.parameterV(i).p3d));
  }
  // binary edge first section
  if (pss.matTypeU() != mat::BlockDiagonal) {
    for (int i = 0; i < numPoints2d; i++) {
      for (int j = i + 1; j < numPoints2d; j++) {
        pss.addEdge(i, j, new EdgeBinaryPoint2d<TestType>(i, j, pss.parameterU(i).p2d - pss.parameterU(j).p2d));
      }
    }
  }
  // binary edge section section
  if (pss.matTypeV() != mat::BlockDiagonal) {
    for (int i = 0; i < numPoints3d; i++) {
      for (int j = i + 1; j < numPoints3d; j++) {
        pss.addEdge(i, j, new EdgeBinaryPoint3d<TestType>(i, j, pss.parameterV(i).p3d - pss.parameterV(j).p3d));
      }
    }
  }
  // binary edge first section to second section
  if (pss.matTypeW() != mat::BlockDiagonal) {
    for (int i = 0; i < numPoints2d; i++) {
      for (int j = 0; j < numPoints3d; j++) {
        pss.addEdge(i, j, new EdgeBinaryPoint2d3d<TestType>(i, j, pss.parameterU(i).p2d - pss.parameterV(j).p3d.template head<2>()));
      }
    }
  }
  else {
    for (int i = 0; i < std::min(numPoints2d, numPoints3d); i++) {
      pss.addEdge(i, i, new EdgeBinaryPoint2d3d<TestType>(i, i, pss.parameterU(i).p2d - pss.parameterV(i).p3d.template head<2>()));
    }
  }


  pss.structureReady();

  pss.update(true);
  std::cout << "chi2 BEFORE " << pss.hessian().chi2() << std::endl;

  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskyDense> > gn;
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskySparse> > gn;
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
  }

  //std::cout << "U " << std::endl << pss.hessian().U().mat() << std::endl << std::endl;
  //std::cout << "V " << std::endl << pss.hessian().V().mat() << std::endl << std::endl;
  //std::cout << "W " << std::endl << pss.hessian().W().mat() << std::endl << std::endl;

}