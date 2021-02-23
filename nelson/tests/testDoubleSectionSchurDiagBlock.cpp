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

constexpr int Point2d::blockSize;

struct Point3d {
  Eigen::Vector3d p3d;
  static constexpr int blockSize = 3;
};

constexpr int Point3d::blockSize;

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

    H += Eigen::Vector3d(1, 1, 0).asDiagonal();
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


template<class Derived, int matTypeUv, int matTypeVv, int matTypeWv, int BU, int BV, int NBU, int NBV>
class Points2d3dBase : public nelson::DoubleSection< Derived, Point2d, Point3d, matTypeUv, matTypeVv, matTypeWv, double, BU, BV, NBU, NBV> {

  std::array<Point2d, numPoints2d> _points2d, _bck_points2d;
  std::array<Point3d, numPoints3d> _points3d, _bck_points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;


public:
  using DoubleSectionBase = nelson::DoubleSection< Derived, Point2d, Point3d, matTypeUv, matTypeVv, matTypeWv, double, BU, BV, NBU, NBV>;

  Points2d3dBase() {
    Eigen::Matrix3Xd groundThruthPoints;
    groundThruthPoints.setRandom(3, std::max(numPoints2d, numPoints3d));

    for (int i = 0; i < numPoints2d; i++) { _points2d[i].p2d = groundThruthPoints.col(i).template head<2>(); }
    for (int i = 0; i < numPoints3d; i++) { _points3d[i].p3d = groundThruthPoints.col(i); }
    _fixedPoint2d.p2d.setRandom();
    _fixedPoint3d.p3d.setRandom();

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

  void backupSolution() {
    _bck_points2d = _points2d;
    _bck_points3d = _points3d;
  }

  void rollbackSolution() {
    _points2d = _bck_points2d;
    _points3d = _bck_points3d;
  }


};

template<class TestType, template<typename> class Solver>
void testFunction() {
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  pss.parametersReady();
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
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverse>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperDense, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering> >;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }

  {
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverse>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperDense, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering> >;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverse>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperSparse, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering> >;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverse>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperSparse, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering> >;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }

}

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

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFFD : public Points2d3dBase < Points2d3dFFFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, numPoints2d, mat::Dynamic > {

public:

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionFFFD_DenseDiagoDense = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DenseDiagoDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseDiagoSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DenseDiagoSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DiagoDiagoDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DiagoDiagoDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoDiagoSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DiagoDiagoSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SparsDiagoDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SparsDiagoDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsDiagoSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SparsDiagoSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SpacoDiagoDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SpacoDiagoDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoDiagoSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SpacoDiagoSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFDF : public Points2d3dBase < Points2d3dFFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, numPoints3d > {

public:

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFFDF_DenseDiagoDense = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DenseDiagoDiago = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DenseDiagoSpars = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DenseDiagoSpaco = Points2d3dFFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_DiagoDiagoDense = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_DiagoDiagoDiago = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_DiagoDiagoSpars = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_DiagoDiagoSpaco = Points2d3dFFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SparsDiagoDense = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SparsDiagoDiago = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SparsDiagoSpars = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SparsDiagoSpaco = Points2d3dFFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDF_SpacoDiagoDense = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDF_SpacoDiagoDiago = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDF_SpacoDiagoSpars = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDF_SpacoDiagoSpaco = Points2d3dFFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFDD : public Points2d3dBase < Points2d3dFFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

public:

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFFDD_DenseDiagoDense = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DenseDiagoDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseDiagoSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DenseDiagoSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoDiagoDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DiagoDiagoDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoDiagoSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DiagoDiagoSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsDiagoDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SparsDiagoDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsDiagoSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SparsDiagoSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoDiagoDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SpacoDiagoDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoDiagoSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SpacoDiagoSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDFF : public Points2d3dBase < Points2d3dFDFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, numPoints2d, numPoints3d > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }


};

using PointsSectionFDFF_DenseDiagoDense = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_DenseDiagoDiago = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_DenseDiagoSpars = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_DenseDiagoSpaco = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DiagoDiagoDense = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_DiagoDiagoDiago = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_DiagoDiagoSpars = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_DiagoDiagoSpaco = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SparsDiagoDense = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_SparsDiagoDiago = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_SparsDiagoSpars = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_SparsDiagoSpaco = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SpacoDiagoDense = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_SpacoDiagoDiago = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_SpacoDiagoSpars = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_SpacoDiagoSpaco = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDFD : public Points2d3dBase < Points2d3dFDFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, numPoints2d, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionFDFD_DenseDiagoDense = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DenseDiagoDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseDiagoSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DenseDiagoSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DiagoDiagoDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DiagoDiagoDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoDiagoSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DiagoDiagoSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SparsDiagoDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SparsDiagoDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsDiagoSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SparsDiagoSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SpacoDiagoDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SpacoDiagoDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoDiagoSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SpacoDiagoSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDDF : public Points2d3dBase < Points2d3dFDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, mat::Dynamic, numPoints3d > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFDDF_DenseDiagoDense = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DenseDiagoDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseDiagoSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DenseDiagoSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DiagoDiagoDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DiagoDiagoDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoDiagoSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DiagoDiagoSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SparsDiagoDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SparsDiagoDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsDiagoSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SparsDiagoSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SpacoDiagoDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SpacoDiagoDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoDiagoSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SpacoDiagoSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDDD : public Points2d3dBase < Points2d3dFDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFDDD_DenseDiagoDense = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DenseDiagoDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseDiagoSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DenseDiagoSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DiagoDiagoDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DiagoDiagoDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoDiagoSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DiagoDiagoSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SparsDiagoDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SparsDiagoDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsDiagoSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SparsDiagoSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SpacoDiagoDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SpacoDiagoDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoDiagoSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SpacoDiagoSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVFF : public Points2d3dBase < Points2d3dFVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVFF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionFVFF_DenseDiagoDense = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_DenseDiagoDiago = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_DenseDiagoSpars = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_DenseDiagoSpaco = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_DiagoDiagoDense = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_DiagoDiagoDiago = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_DiagoDiagoSpars = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_DiagoDiagoSpaco = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_SparsDiagoDense = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_SparsDiagoDiago = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_SparsDiagoSpars = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_SparsDiagoSpaco = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_SpacoDiagoDense = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_SpacoDiagoDiago = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_SpacoDiagoSpars = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_SpacoDiagoSpaco = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVFD : public Points2d3dBase < Points2d3dFVFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, numPoints2d, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVFD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionFVFD_DenseDiagoDense = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_DenseDiagoDiago = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_DenseDiagoSpars = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_DenseDiagoSpaco = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_DiagoDiagoDense = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_DiagoDiagoDiago = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_DiagoDiagoSpars = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_DiagoDiagoSpaco = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_SparsDiagoDense = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_SparsDiagoDiago = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_SparsDiagoSpars = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_SparsDiagoSpaco = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_SpacoDiagoDense = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_SpacoDiagoDiago = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_SpacoDiagoSpars = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_SpacoDiagoSpaco = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVDF : public Points2d3dBase < Points2d3dFVDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, mat::Dynamic, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVDF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFVDF_DenseDiagoDense = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_DenseDiagoDiago = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_DenseDiagoSpars = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_DenseDiagoSpaco = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_DiagoDiagoDense = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_DiagoDiagoDiago = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_DiagoDiagoSpars = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_DiagoDiagoSpaco = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_SparsDiagoDense = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_SparsDiagoDiago = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_SparsDiagoSpars = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_SparsDiagoSpaco = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_SpacoDiagoDense = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_SpacoDiagoDiago = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_SpacoDiagoSpars = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_SpacoDiagoSpaco = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVDD : public Points2d3dBase < Points2d3dFVDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, mat::Dynamic, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVDD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFVDD_DenseDiagoDense = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_DenseDiagoDiago = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_DenseDiagoSpars = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_DenseDiagoSpaco = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_DiagoDiagoDense = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_DiagoDiagoDiago = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_DiagoDiagoSpars = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_DiagoDiagoSpaco = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_SparsDiagoDense = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_SparsDiagoDiago = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_SparsDiagoSpars = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_SparsDiagoSpaco = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_SpacoDiagoDense = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_SpacoDiagoDiago = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_SpacoDiagoSpars = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_SpacoDiagoSpaco = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFFF : public Points2d3dBase < Points2d3dDFFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, numPoints2d, numPoints3d > {

public:
  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }


};

using PointsSectionDFFF_DenseDiagoDense = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_DenseDiagoDiago = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_DenseDiagoSpars = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_DenseDiagoSpaco = Points2d3dDFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_DiagoDiagoDense = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_DiagoDiagoDiago = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_DiagoDiagoSpars = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_DiagoDiagoSpaco = Points2d3dDFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_SparsDiagoDense = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_SparsDiagoDiago = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_SparsDiagoSpars = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_SparsDiagoSpaco = Points2d3dDFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFF_SpacoDiagoDense = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFF_SpacoDiagoDiago = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFF_SpacoDiagoSpars = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFF_SpacoDiagoSpaco = Points2d3dDFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFFD : public Points2d3dBase < Points2d3dDFFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, numPoints2d, mat::Dynamic > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionDFFD_DenseDiagoDense = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_DenseDiagoDiago = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_DenseDiagoSpars = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_DenseDiagoSpaco = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_DiagoDiagoDense = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_DiagoDiagoDiago = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_DiagoDiagoSpars = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_DiagoDiagoSpaco = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_SparsDiagoDense = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_SparsDiagoDiago = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_SparsDiagoSpars = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_SparsDiagoSpaco = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_SpacoDiagoDense = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_SpacoDiagoDiago = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_SpacoDiagoSpars = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_SpacoDiagoSpaco = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFDF : public Points2d3dBase < Points2d3dDFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, mat::Dynamic, numPoints3d > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDFDF_DenseDiagoDense = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_DenseDiagoDiago = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_DenseDiagoSpars = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_DenseDiagoSpaco = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_DiagoDiagoDense = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_DiagoDiagoDiago = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_DiagoDiagoSpars = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_DiagoDiagoSpaco = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_SparsDiagoDense = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_SparsDiagoDiago = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_SparsDiagoSpars = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_SparsDiagoSpaco = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_SpacoDiagoDense = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_SpacoDiagoDiago = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_SpacoDiagoSpars = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_SpacoDiagoSpaco = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFDD : public Points2d3dBase < Points2d3dDFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDFDD_DenseDiagoDense = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_DenseDiagoDiago = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_DenseDiagoSpars = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_DenseDiagoSpaco = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_DiagoDiagoDense = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_DiagoDiagoDiago = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_DiagoDiagoSpars = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_DiagoDiagoSpaco = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_SparsDiagoDense = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_SparsDiagoDiago = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_SparsDiagoSpars = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_SparsDiagoSpaco = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_SpacoDiagoDense = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_SpacoDiagoDiago = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_SpacoDiagoSpars = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_SpacoDiagoSpaco = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDFF : public Points2d3dBase < Points2d3dDDFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, numPoints2d, numPoints3d > {

public:
  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }


};

using PointsSectionDDFF_DenseDiagoDense = Points2d3dDDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFF_DenseDiagoDiago = Points2d3dDDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFF_DenseDiagoSpars = Points2d3dDDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFF_DenseDiagoSpaco = Points2d3dDDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFF_DiagoDiagoDense = Points2d3dDDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFF_DiagoDiagoDiago = Points2d3dDDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFF_DiagoDiagoSpars = Points2d3dDDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFF_DiagoDiagoSpaco = Points2d3dDDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFF_SparsDiagoDense = Points2d3dDDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFF_SparsDiagoDiago = Points2d3dDDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFF_SparsDiagoSpars = Points2d3dDDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFF_SparsDiagoSpaco = Points2d3dDDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFF_SpacoDiagoDense = Points2d3dDDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFF_SpacoDiagoDiago = Points2d3dDDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFF_SpacoDiagoSpars = Points2d3dDDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFF_SpacoDiagoSpaco = Points2d3dDDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDFD : public Points2d3dBase < Points2d3dDDFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, numPoints2d, mat::Dynamic > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionDDFD_DenseDiagoDense = Points2d3dDDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFD_DenseDiagoDiago = Points2d3dDDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFD_DenseDiagoSpars = Points2d3dDDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFD_DenseDiagoSpaco = Points2d3dDDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFD_DiagoDiagoDense = Points2d3dDDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFD_DiagoDiagoDiago = Points2d3dDDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFD_DiagoDiagoSpars = Points2d3dDDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFD_DiagoDiagoSpaco = Points2d3dDDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFD_SparsDiagoDense = Points2d3dDDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFD_SparsDiagoDiago = Points2d3dDDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFD_SparsDiagoSpars = Points2d3dDDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFD_SparsDiagoSpaco = Points2d3dDDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDFD_SpacoDiagoDense = Points2d3dDDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDFD_SpacoDiagoDiago = Points2d3dDDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDFD_SpacoDiagoSpars = Points2d3dDDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDFD_SpacoDiagoSpaco = Points2d3dDDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDDF : public Points2d3dBase < Points2d3dDDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, mat::Dynamic, numPoints3d > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDDDF_DenseDiagoDense = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_DenseDiagoDiago = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_DenseDiagoSpars = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_DenseDiagoSpaco = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_DiagoDiagoDense = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_DiagoDiagoDiago = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_DiagoDiagoSpars = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_DiagoDiagoSpaco = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_SparsDiagoDense = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_SparsDiagoDiago = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_SparsDiagoSpars = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_SparsDiagoSpaco = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_SpacoDiagoDense = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_SpacoDiagoDiago = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_SpacoDiagoSpars = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_SpacoDiagoSpaco = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDDD : public Points2d3dBase < Points2d3dDDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDDDD_DenseDiagoDense = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_DenseDiagoDiago = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_DenseDiagoSpars = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_DenseDiagoSpaco = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_DiagoDiagoDense = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_DiagoDiagoDiago = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_DiagoDiagoSpars = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_DiagoDiagoSpaco = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_SparsDiagoDense = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_SparsDiagoDiago = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_SparsDiagoSpars = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_SparsDiagoSpaco = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_SpacoDiagoDense = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_SpacoDiagoDiago = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_SpacoDiagoSpars = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_SpacoDiagoSpaco = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVFF : public Points2d3dBase < Points2d3dDVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVFF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionDVFF_DenseDiagoDense = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_DenseDiagoDiago = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_DenseDiagoSpars = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_DenseDiagoSpaco = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_DiagoDiagoDense = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_DiagoDiagoDiago = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_DiagoDiagoSpars = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_DiagoDiagoSpaco = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_SparsDiagoDense = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_SparsDiagoDiago = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_SparsDiagoSpars = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_SparsDiagoSpaco = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_SpacoDiagoDense = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_SpacoDiagoDiago = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_SpacoDiagoSpars = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_SpacoDiagoSpaco = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVFD : public Points2d3dBase< Points2d3dDVFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, numPoints2d, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVFD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionDVFD_DenseDiagoDense = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_DenseDiagoDiago = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_DenseDiagoSpars = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_DenseDiagoSpaco = Points2d3dDVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_DiagoDiagoDense = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_DiagoDiagoDiago = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_DiagoDiagoSpars = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_DiagoDiagoSpaco = Points2d3dDVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_SparsDiagoDense = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_SparsDiagoDiago = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_SparsDiagoSpars = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_SparsDiagoSpaco = Points2d3dDVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFD_SpacoDiagoDense = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFD_SpacoDiagoDiago = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFD_SpacoDiagoSpars = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFD_SpacoDiagoSpaco = Points2d3dDVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

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

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVDD : public Points2d3dBase < Points2d3dDVDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVDD() : v_sizes(numPoints3d, Point3d::blockSize) { }

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

using PointsSectionDVDD_DenseDiagoDense = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_DenseDiagoDiago = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_DenseDiagoSpars = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_DenseDiagoSpaco = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_DiagoDiagoDense = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_DiagoDiagoDiago = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_DiagoDiagoSpars = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_DiagoDiagoSpaco = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_SparsDiagoDense = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_SparsDiagoDiago = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_SparsDiagoSpars = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_SparsDiagoSpaco = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_SpacoDiagoDense = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_SpacoDiagoDiago = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_SpacoDiagoSpars = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_SpacoDiagoSpaco = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFFF : public Points2d3dBase < Points2d3dVFFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, numPoints2d, numPoints3d > {
  std::vector<int> u_sizes;
public:

  Points2d3dVFFF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }


};

using PointsSectionVFFF_DenseDiagoDense = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_DenseDiagoDiago = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_DenseDiagoSpars = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_DenseDiagoSpaco = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_DiagoDiagoDense = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_DiagoDiagoDiago = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_DiagoDiagoSpars = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_DiagoDiagoSpaco = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_SparsDiagoDense = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_SparsDiagoDiago = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_SparsDiagoSpars = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_SparsDiagoSpaco = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_SpacoDiagoDense = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_SpacoDiagoDiago = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_SpacoDiagoSpars = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_SpacoDiagoSpaco = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFFD : public Points2d3dBase < Points2d3dVFFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, numPoints2d, mat::Dynamic > {
  std::vector<int> u_sizes;
public:

  Points2d3dVFFD() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }
  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionVFFD_DenseDiagoDense = Points2d3dVFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFD_DenseDiagoDiago = Points2d3dVFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFD_DenseDiagoSpars = Points2d3dVFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFD_DenseDiagoSpaco = Points2d3dVFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFD_DiagoDiagoDense = Points2d3dVFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFD_DiagoDiagoDiago = Points2d3dVFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFD_DiagoDiagoSpars = Points2d3dVFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFD_DiagoDiagoSpaco = Points2d3dVFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFD_SparsDiagoDense = Points2d3dVFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFD_SparsDiagoDiago = Points2d3dVFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFD_SparsDiagoSpars = Points2d3dVFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFD_SparsDiagoSpaco = Points2d3dVFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFD_SpacoDiagoDense = Points2d3dVFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFD_SpacoDiagoDiago = Points2d3dVFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFD_SpacoDiagoSpars = Points2d3dVFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFD_SpacoDiagoSpaco = Points2d3dVFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFDF : public Points2d3dBase < Points2d3dVFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, mat::Dynamic, numPoints3d > {

  std::vector<int> u_sizes;
public:

  Points2d3dVFDF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

};

using PointsSectionVFDF_DenseDiagoDense = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_DenseDiagoDiago = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_DenseDiagoSpars = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_DenseDiagoSpaco = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_DiagoDiagoDense = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_DiagoDiagoDiago = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_DiagoDiagoSpars = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_DiagoDiagoSpaco = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_SparsDiagoDense = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_SparsDiagoDiago = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_SparsDiagoSpars = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_SparsDiagoSpaco = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_SpacoDiagoDense = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_SpacoDiagoDiago = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_SpacoDiagoSpars = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_SpacoDiagoSpaco = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFDD : public Points2d3dBase < Points2d3dVFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

  std::vector<int> u_sizes;
public:

  Points2d3dVFDD() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionVFDD_DenseDiagoDense = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_DenseDiagoDiago = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_DenseDiagoSpars = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_DenseDiagoSpaco = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_DiagoDiagoDense = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_DiagoDiagoDiago = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_DiagoDiagoSpars = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_DiagoDiagoSpaco = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_SparsDiagoDense = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_SparsDiagoDiago = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_SparsDiagoSpars = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_SparsDiagoSpaco = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_SpacoDiagoDense = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_SpacoDiagoDiago = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_SpacoDiagoSpars = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_SpacoDiagoSpaco = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDFF : public Points2d3dBase < Points2d3dVDFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, numPoints2d, numPoints3d > {

  std::vector<int> u_sizes;
public:

  Points2d3dVDFF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }


};

using PointsSectionVDFF_DenseDiagoDense = Points2d3dVDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFF_DenseDiagoDiago = Points2d3dVDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFF_DenseDiagoSpars = Points2d3dVDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFF_DenseDiagoSpaco = Points2d3dVDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFF_DiagoDiagoDense = Points2d3dVDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFF_DiagoDiagoDiago = Points2d3dVDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFF_DiagoDiagoSpars = Points2d3dVDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFF_DiagoDiagoSpaco = Points2d3dVDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFF_SparsDiagoDense = Points2d3dVDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFF_SparsDiagoDiago = Points2d3dVDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFF_SparsDiagoSpars = Points2d3dVDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFF_SparsDiagoSpaco = Points2d3dVDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFF_SpacoDiagoDense = Points2d3dVDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFF_SpacoDiagoDiago = Points2d3dVDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFF_SpacoDiagoSpars = Points2d3dVDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFF_SpacoDiagoSpaco = Points2d3dVDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDFD : public Points2d3dBase < Points2d3dVDFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, numPoints2d, mat::Dynamic > {
  std::vector<int> u_sizes;
public:

  Points2d3dVDFD() : u_sizes(numPoints2d, Point2d::blockSize) { }
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

using PointsSectionVDFD_DenseDiagoDense = Points2d3dVDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFD_DenseDiagoDiago = Points2d3dVDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFD_DenseDiagoSpars = Points2d3dVDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFD_DenseDiagoSpaco = Points2d3dVDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFD_DiagoDiagoDense = Points2d3dVDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFD_DiagoDiagoDiago = Points2d3dVDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFD_DiagoDiagoSpars = Points2d3dVDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFD_DiagoDiagoSpaco = Points2d3dVDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFD_SparsDiagoDense = Points2d3dVDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFD_SparsDiagoDiago = Points2d3dVDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFD_SparsDiagoSpars = Points2d3dVDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFD_SparsDiagoSpaco = Points2d3dVDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDFD_SpacoDiagoDense = Points2d3dVDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDFD_SpacoDiagoDiago = Points2d3dVDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDFD_SpacoDiagoSpars = Points2d3dVDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDFD_SpacoDiagoSpaco = Points2d3dVDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDDF : public Points2d3dBase < Points2d3dVDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, mat::Dynamic, numPoints3d > {

  std::vector<int> u_sizes;
public:

  Points2d3dVDDF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }
  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

};

using PointsSectionVDDF_DenseDiagoDense = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_DenseDiagoDiago = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_DenseDiagoSpars = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_DenseDiagoSpaco = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_DiagoDiagoDense = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_DiagoDiagoDiago = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_DiagoDiagoSpars = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_DiagoDiagoSpaco = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_SparsDiagoDense = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_SparsDiagoDiago = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_SparsDiagoSpars = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_SparsDiagoSpaco = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_SpacoDiagoDense = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_SpacoDiagoDiago = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_SpacoDiagoSpars = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_SpacoDiagoSpaco = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDDD : public Points2d3dBase < Points2d3dVDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

  std::vector<int> u_sizes;
public:

  Points2d3dVDDD() : u_sizes(numPoints2d, Point2d::blockSize) { }
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

using PointsSectionVDDD_DenseDiagoDense = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_DenseDiagoDiago = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_DenseDiagoSpars = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_DenseDiagoSpaco = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_DiagoDiagoDense = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_DiagoDiagoDiago = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_DiagoDiagoSpars = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_DiagoDiagoSpaco = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_SparsDiagoDense = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_SparsDiagoDiago = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_SparsDiagoSpars = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_SparsDiagoSpaco = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_SpacoDiagoDense = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_SpacoDiagoDiago = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_SpacoDiagoSpars = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_SpacoDiagoSpaco = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVVFF : public Points2d3dBase < Points2d3dVVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
  std::vector<int> u_sizes;
public:

  Points2d3dVVFF() : v_sizes(numPoints3d, Point3d::blockSize), u_sizes(numPoints2d, Point2d::blockSize) { }

  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionVVFF_DenseDiagoDense = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_DenseDiagoDiago = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_DenseDiagoSpars = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_DenseDiagoSpaco = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_DiagoDiagoDense = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_DiagoDiagoDiago = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_DiagoDiagoSpars = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_DiagoDiagoSpaco = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_SparsDiagoDense = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_SparsDiagoDiago = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_SparsDiagoSpars = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_SparsDiagoSpaco = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_SpacoDiagoDense = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_SpacoDiagoDiago = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_SpacoDiagoSpars = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_SpacoDiagoSpaco = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

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


TEMPLATE_TEST_CASE("DoubleSection-DF", "[DoubleSection-DF]",
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
  PointsSectionFFFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFFFD_DenseDiagoDense,
  PointsSectionFFFD_DenseDiagoDiago,
  PointsSectionFFFD_DenseDiagoSpars,
  PointsSectionFFFD_DenseDiagoSpaco,
  PointsSectionFFFD_DiagoDiagoDense,
  PointsSectionFFFD_DiagoDiagoDiago,
  PointsSectionFFFD_DiagoDiagoSpars,
  PointsSectionFFFD_DiagoDiagoSpaco,
  PointsSectionFFFD_SparsDiagoDense,
  PointsSectionFFFD_SparsDiagoDiago,
  PointsSectionFFFD_SparsDiagoSpars,
  PointsSectionFFFD_SparsDiagoSpaco,
  PointsSectionFFFD_SpacoDiagoDense,
  PointsSectionFFFD_SpacoDiagoDiago,
  PointsSectionFFFD_SpacoDiagoSpars,
  PointsSectionFFFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFFDF_DenseDiagoDense,
  PointsSectionFFDF_DenseDiagoDiago,
  PointsSectionFFDF_DenseDiagoSpars,
  PointsSectionFFDF_DenseDiagoSpaco,
  PointsSectionFFDF_DiagoDiagoDense,
  PointsSectionFFDF_DiagoDiagoDiago,
  PointsSectionFFDF_DiagoDiagoSpars,
  PointsSectionFFDF_DiagoDiagoSpaco,
  PointsSectionFFDF_SparsDiagoDense,
  PointsSectionFFDF_SparsDiagoDiago,
  PointsSectionFFDF_SparsDiagoSpars,
  PointsSectionFFDF_SparsDiagoSpaco,
  PointsSectionFFDF_SpacoDiagoDense,
  PointsSectionFFDF_SpacoDiagoDiago,
  PointsSectionFFDF_SpacoDiagoSpars,
  PointsSectionFFDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFFDD_DenseDiagoDense,
  PointsSectionFFDD_DenseDiagoDiago,
  PointsSectionFFDD_DenseDiagoSpars,
  PointsSectionFFDD_DenseDiagoSpaco,
  PointsSectionFFDD_DiagoDiagoDense,
  PointsSectionFFDD_DiagoDiagoDiago,
  PointsSectionFFDD_DiagoDiagoSpars,
  PointsSectionFFDD_DiagoDiagoSpaco,
  PointsSectionFFDD_SparsDiagoDense,
  PointsSectionFFDD_SparsDiagoDiago,
  PointsSectionFFDD_SparsDiagoSpars,
  PointsSectionFFDD_SparsDiagoSpaco,
  PointsSectionFFDD_SpacoDiagoDense,
  PointsSectionFFDD_SpacoDiagoDiago,
  PointsSectionFFDD_SpacoDiagoSpars,
  PointsSectionFFDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFDFF_DenseDiagoDense,
  PointsSectionFDFF_DenseDiagoDiago,
  PointsSectionFDFF_DenseDiagoSpars,
  PointsSectionFDFF_DenseDiagoSpaco,
  PointsSectionFDFF_DiagoDiagoDense,
  PointsSectionFDFF_DiagoDiagoDiago,
  PointsSectionFDFF_DiagoDiagoSpars,
  PointsSectionFDFF_DiagoDiagoSpaco,
  PointsSectionFDFF_SparsDiagoDense,
  PointsSectionFDFF_SparsDiagoDiago,
  PointsSectionFDFF_SparsDiagoSpars,
  PointsSectionFDFF_SparsDiagoSpaco,
  PointsSectionFDFF_SpacoDiagoDense,
  PointsSectionFDFF_SpacoDiagoDiago,
  PointsSectionFDFF_SpacoDiagoSpars,
  PointsSectionFDFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFDFD_DenseDiagoDense,
  PointsSectionFDFD_DenseDiagoDiago,
  PointsSectionFDFD_DenseDiagoSpars,
  PointsSectionFDFD_DenseDiagoSpaco,
  PointsSectionFDFD_DiagoDiagoDense,
  PointsSectionFDFD_DiagoDiagoDiago,
  PointsSectionFDFD_DiagoDiagoSpars,
  PointsSectionFDFD_DiagoDiagoSpaco,
  PointsSectionFDFD_SparsDiagoDense,
  PointsSectionFDFD_SparsDiagoDiago,
  PointsSectionFDFD_SparsDiagoSpars,
  PointsSectionFDFD_SparsDiagoSpaco,
  PointsSectionFDFD_SpacoDiagoDense,
  PointsSectionFDFD_SpacoDiagoDiago,
  PointsSectionFDFD_SpacoDiagoSpars,
  PointsSectionFDFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFDDF_DenseDiagoDense,
  PointsSectionFDDF_DenseDiagoDiago,
  PointsSectionFDDF_DenseDiagoSpars,
  PointsSectionFDDF_DenseDiagoSpaco,
  PointsSectionFDDF_DiagoDiagoDense,
  PointsSectionFDDF_DiagoDiagoDiago,
  PointsSectionFDDF_DiagoDiagoSpars,
  PointsSectionFDDF_DiagoDiagoSpaco,
  PointsSectionFDDF_SparsDiagoDense,
  PointsSectionFDDF_SparsDiagoDiago,
  PointsSectionFDDF_SparsDiagoSpars,
  PointsSectionFDDF_SparsDiagoSpaco,
  PointsSectionFDDF_SpacoDiagoDense,
  PointsSectionFDDF_SpacoDiagoDiago,
  PointsSectionFDDF_SpacoDiagoSpars,
  PointsSectionFDDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFDDD_DenseDiagoDense,
  PointsSectionFDDD_DenseDiagoDiago,
  PointsSectionFDDD_DenseDiagoSpars,
  PointsSectionFDDD_DenseDiagoSpaco,
  PointsSectionFDDD_DiagoDiagoDense,
  PointsSectionFDDD_DiagoDiagoDiago,
  PointsSectionFDDD_DiagoDiagoSpars,
  PointsSectionFDDD_DiagoDiagoSpaco,
  PointsSectionFDDD_SparsDiagoDense,
  PointsSectionFDDD_SparsDiagoDiago,
  PointsSectionFDDD_SparsDiagoSpars,
  PointsSectionFDDD_SparsDiagoSpaco,
  PointsSectionFDDD_SpacoDiagoDense,
  PointsSectionFDDD_SpacoDiagoDiago,
  PointsSectionFDDD_SpacoDiagoSpars,
  PointsSectionFDDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDFFF_DenseDiagoDense,
  PointsSectionDFFF_DenseDiagoDiago,
  PointsSectionDFFF_DenseDiagoSpars,
  PointsSectionDFFF_DenseDiagoSpaco,
  PointsSectionDFFF_DiagoDiagoDense,
  PointsSectionDFFF_DiagoDiagoDiago,
  PointsSectionDFFF_DiagoDiagoSpars,
  PointsSectionDFFF_DiagoDiagoSpaco,
  PointsSectionDFFF_SparsDiagoDense,
  PointsSectionDFFF_SparsDiagoDiago,
  PointsSectionDFFF_SparsDiagoSpars,
  PointsSectionDFFF_SparsDiagoSpaco,
  PointsSectionDFFF_SpacoDiagoDense,
  PointsSectionDFFF_SpacoDiagoDiago,
  PointsSectionDFFF_SpacoDiagoSpars,
  PointsSectionDFFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDFFD_DenseDiagoDense,
  PointsSectionDFFD_DenseDiagoDiago,
  PointsSectionDFFD_DenseDiagoSpars,
  PointsSectionDFFD_DenseDiagoSpaco,
  PointsSectionDFFD_DiagoDiagoDense,
  PointsSectionDFFD_DiagoDiagoDiago,
  PointsSectionDFFD_DiagoDiagoSpars,
  PointsSectionDFFD_DiagoDiagoSpaco,
  PointsSectionDFFD_SparsDiagoDense,
  PointsSectionDFFD_SparsDiagoDiago,
  PointsSectionDFFD_SparsDiagoSpars,
  PointsSectionDFFD_SparsDiagoSpaco,
  PointsSectionDFFD_SpacoDiagoDense,
  PointsSectionDFFD_SpacoDiagoDiago,
  PointsSectionDFFD_SpacoDiagoSpars,
  PointsSectionDFFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDFDF_DenseDiagoDense,
  PointsSectionDFDF_DenseDiagoDiago,
  PointsSectionDFDF_DenseDiagoSpars,
  PointsSectionDFDF_DenseDiagoSpaco,
  PointsSectionDFDF_DiagoDiagoDense,
  PointsSectionDFDF_DiagoDiagoDiago,
  PointsSectionDFDF_DiagoDiagoSpars,
  PointsSectionDFDF_DiagoDiagoSpaco,
  PointsSectionDFDF_SparsDiagoDense,
  PointsSectionDFDF_SparsDiagoDiago,
  PointsSectionDFDF_SparsDiagoSpars,
  PointsSectionDFDF_SparsDiagoSpaco,
  PointsSectionDFDF_SpacoDiagoDense,
  PointsSectionDFDF_SpacoDiagoDiago,
  PointsSectionDFDF_SpacoDiagoSpars,
  PointsSectionDFDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDFDD_DenseDiagoDense,
  PointsSectionDFDD_DenseDiagoDiago,
  PointsSectionDFDD_DenseDiagoSpars,
  PointsSectionDFDD_DenseDiagoSpaco,
  PointsSectionDFDD_DiagoDiagoDense,
  PointsSectionDFDD_DiagoDiagoDiago,
  PointsSectionDFDD_DiagoDiagoSpars,
  PointsSectionDFDD_DiagoDiagoSpaco,
  PointsSectionDFDD_SparsDiagoDense,
  PointsSectionDFDD_SparsDiagoDiago,
  PointsSectionDFDD_SparsDiagoSpars,
  PointsSectionDFDD_SparsDiagoSpaco,
  PointsSectionDFDD_SpacoDiagoDense,
  PointsSectionDFDD_SpacoDiagoDiago,
  PointsSectionDFDD_SpacoDiagoSpars,
  PointsSectionDFDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDDFF_DenseDiagoDense,
  PointsSectionDDFF_DenseDiagoDiago,
  PointsSectionDDFF_DenseDiagoSpars,
  PointsSectionDDFF_DenseDiagoSpaco,
  PointsSectionDDFF_DiagoDiagoDense,
  PointsSectionDDFF_DiagoDiagoDiago,
  PointsSectionDDFF_DiagoDiagoSpars,
  PointsSectionDDFF_DiagoDiagoSpaco,
  PointsSectionDDFF_SparsDiagoDense,
  PointsSectionDDFF_SparsDiagoDiago,
  PointsSectionDDFF_SparsDiagoSpars,
  PointsSectionDDFF_SparsDiagoSpaco,
  PointsSectionDDFF_SpacoDiagoDense,
  PointsSectionDDFF_SpacoDiagoDiago,
  PointsSectionDDFF_SpacoDiagoSpars,
  PointsSectionDDFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDDFD_DenseDiagoDense,
  PointsSectionDDFD_DenseDiagoDiago,
  PointsSectionDDFD_DenseDiagoSpars,
  PointsSectionDDFD_DenseDiagoSpaco,
  PointsSectionDDFD_DiagoDiagoDense,
  PointsSectionDDFD_DiagoDiagoDiago,
  PointsSectionDDFD_DiagoDiagoSpars,
  PointsSectionDDFD_DiagoDiagoSpaco,
  PointsSectionDDFD_SparsDiagoDense,
  PointsSectionDDFD_SparsDiagoDiago,
  PointsSectionDDFD_SparsDiagoSpars,
  PointsSectionDDFD_SparsDiagoSpaco,
  PointsSectionDDFD_SpacoDiagoDense,
  PointsSectionDDFD_SpacoDiagoDiago,
  PointsSectionDDFD_SpacoDiagoSpars,
  PointsSectionDDFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDDDF_DenseDiagoDense,
  PointsSectionDDDF_DenseDiagoDiago,
  PointsSectionDDDF_DenseDiagoSpars,
  PointsSectionDDDF_DenseDiagoSpaco,
  PointsSectionDDDF_DiagoDiagoDense,
  PointsSectionDDDF_DiagoDiagoDiago,
  PointsSectionDDDF_DiagoDiagoSpars,
  PointsSectionDDDF_DiagoDiagoSpaco,
  PointsSectionDDDF_SparsDiagoDense,
  PointsSectionDDDF_SparsDiagoDiago,
  PointsSectionDDDF_SparsDiagoSpars,
  PointsSectionDDDF_SparsDiagoSpaco,
  PointsSectionDDDF_SpacoDiagoDense,
  PointsSectionDDDF_SpacoDiagoDiago,
  PointsSectionDDDF_SpacoDiagoSpars,
  PointsSectionDDDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDDDD_DenseDiagoDense,
  PointsSectionDDDD_DenseDiagoDiago,
  PointsSectionDDDD_DenseDiagoSpars,
  PointsSectionDDDD_DenseDiagoSpaco,
  PointsSectionDDDD_DiagoDiagoDense,
  PointsSectionDDDD_DiagoDiagoDiago,
  PointsSectionDDDD_DiagoDiagoSpars,
  PointsSectionDDDD_DiagoDiagoSpaco,
  PointsSectionDDDD_SparsDiagoDense,
  PointsSectionDDDD_SparsDiagoDiago,
  PointsSectionDDDD_SparsDiagoSpars,
  PointsSectionDDDD_SparsDiagoSpaco,
  PointsSectionDDDD_SpacoDiagoDense,
  PointsSectionDDDD_SpacoDiagoDiago,
  PointsSectionDDDD_SpacoDiagoSpars,
  PointsSectionDDDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

TEMPLATE_TEST_CASE("DoubleSection-DF", "[DoubleSection-DF]",
  //----------------------------------
  PointsSectionFVFF_DenseDiagoDense,
  PointsSectionFVFF_DenseDiagoDiago,
  PointsSectionFVFF_DenseDiagoSpars,
  PointsSectionFVFF_DenseDiagoSpaco,
  PointsSectionFVFF_DiagoDiagoDense,
  PointsSectionFVFF_DiagoDiagoDiago,
  PointsSectionFVFF_DiagoDiagoSpars,
  PointsSectionFVFF_DiagoDiagoSpaco,
  PointsSectionFVFF_SparsDiagoDense,
  PointsSectionFVFF_SparsDiagoDiago,
  PointsSectionFVFF_SparsDiagoSpars,
  PointsSectionFVFF_SparsDiagoSpaco,
  PointsSectionFVFF_SpacoDiagoDense,
  PointsSectionFVFF_SpacoDiagoDiago,
  PointsSectionFVFF_SpacoDiagoSpars,
  PointsSectionFVFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFVFD_DenseDiagoDense,
  PointsSectionFVFD_DenseDiagoDiago,
  PointsSectionFVFD_DenseDiagoSpars,
  PointsSectionFVFD_DenseDiagoSpaco,
  PointsSectionFVFD_DiagoDiagoDense,
  PointsSectionFVFD_DiagoDiagoDiago,
  PointsSectionFVFD_DiagoDiagoSpars,
  PointsSectionFVFD_DiagoDiagoSpaco,
  PointsSectionFVFD_SparsDiagoDense,
  PointsSectionFVFD_SparsDiagoDiago,
  PointsSectionFVFD_SparsDiagoSpars,
  PointsSectionFVFD_SparsDiagoSpaco,
  PointsSectionFVFD_SpacoDiagoDense,
  PointsSectionFVFD_SpacoDiagoDiago,
  PointsSectionFVFD_SpacoDiagoSpars,
  PointsSectionFVFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFVDF_DenseDiagoDense,
  PointsSectionFVDF_DenseDiagoDiago,
  PointsSectionFVDF_DenseDiagoSpars,
  PointsSectionFVDF_DenseDiagoSpaco,
  PointsSectionFVDF_DiagoDiagoDense,
  PointsSectionFVDF_DiagoDiagoDiago,
  PointsSectionFVDF_DiagoDiagoSpars,
  PointsSectionFVDF_DiagoDiagoSpaco,
  PointsSectionFVDF_SparsDiagoDense,
  PointsSectionFVDF_SparsDiagoDiago,
  PointsSectionFVDF_SparsDiagoSpars,
  PointsSectionFVDF_SparsDiagoSpaco,
  PointsSectionFVDF_SpacoDiagoDense,
  PointsSectionFVDF_SpacoDiagoDiago,
  PointsSectionFVDF_SpacoDiagoSpars,
  PointsSectionFVDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionFVDD_DenseDiagoDense,
  PointsSectionFVDD_DenseDiagoDiago,
  PointsSectionFVDD_DenseDiagoSpars,
  PointsSectionFVDD_DenseDiagoSpaco,
  PointsSectionFVDD_DiagoDiagoDense,
  PointsSectionFVDD_DiagoDiagoDiago,
  PointsSectionFVDD_DiagoDiagoSpars,
  PointsSectionFVDD_DiagoDiagoSpaco,
  PointsSectionFVDD_SparsDiagoDense,
  PointsSectionFVDD_SparsDiagoDiago,
  PointsSectionFVDD_SparsDiagoSpars,
  PointsSectionFVDD_SparsDiagoSpaco,
  PointsSectionFVDD_SpacoDiagoDense,
  PointsSectionFVDD_SpacoDiagoDiago,
  PointsSectionFVDD_SpacoDiagoSpars,
  PointsSectionFVDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDVFF_DenseDiagoDense,
  PointsSectionDVFF_DenseDiagoDiago,
  PointsSectionDVFF_DenseDiagoSpars,
  PointsSectionDVFF_DenseDiagoSpaco,
  PointsSectionDVFF_DiagoDiagoDense,
  PointsSectionDVFF_DiagoDiagoDiago,
  PointsSectionDVFF_DiagoDiagoSpars,
  PointsSectionDVFF_DiagoDiagoSpaco,
  PointsSectionDVFF_SparsDiagoDense,
  PointsSectionDVFF_SparsDiagoDiago,
  PointsSectionDVFF_SparsDiagoSpars,
  PointsSectionDVFF_SparsDiagoSpaco,
  PointsSectionDVFF_SpacoDiagoDense,
  PointsSectionDVFF_SpacoDiagoDiago,
  PointsSectionDVFF_SpacoDiagoSpars,
  PointsSectionDVFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDVFD_DenseDiagoDense,
  PointsSectionDVFD_DenseDiagoDiago,
  PointsSectionDVFD_DenseDiagoSpars,
  PointsSectionDVFD_DenseDiagoSpaco,
  PointsSectionDVFD_DiagoDiagoDense,
  PointsSectionDVFD_DiagoDiagoDiago,
  PointsSectionDVFD_DiagoDiagoSpars,
  PointsSectionDVFD_DiagoDiagoSpaco,
  PointsSectionDVFD_SparsDiagoDense,
  PointsSectionDVFD_SparsDiagoDiago,
  PointsSectionDVFD_SparsDiagoSpars,
  PointsSectionDVFD_SparsDiagoSpaco,
  PointsSectionDVFD_SpacoDiagoDense,
  PointsSectionDVFD_SpacoDiagoDiago,
  PointsSectionDVFD_SpacoDiagoSpars,
  PointsSectionDVFD_SpacoDiagoSpaco,
  //----------------------------------
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
  PointsSectionDVDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionDVDD_DenseDiagoDense,
  PointsSectionDVDD_DenseDiagoDiago,
  PointsSectionDVDD_DenseDiagoSpars,
  PointsSectionDVDD_DenseDiagoSpaco,
  PointsSectionDVDD_DiagoDiagoDense,
  PointsSectionDVDD_DiagoDiagoDiago,
  PointsSectionDVDD_DiagoDiagoSpars,
  PointsSectionDVDD_DiagoDiagoSpaco,
  PointsSectionDVDD_SparsDiagoDense,
  PointsSectionDVDD_SparsDiagoDiago,
  PointsSectionDVDD_SparsDiagoSpars,
  PointsSectionDVDD_SparsDiagoSpaco,
  PointsSectionDVDD_SpacoDiagoDense,
  PointsSectionDVDD_SpacoDiagoDiago,
  PointsSectionDVDD_SpacoDiagoSpars,
  PointsSectionDVDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVFFF_DenseDiagoDense,
  PointsSectionVFFF_DenseDiagoDiago,
  PointsSectionVFFF_DenseDiagoSpars,
  PointsSectionVFFF_DenseDiagoSpaco,
  PointsSectionVFFF_DiagoDiagoDense,
  PointsSectionVFFF_DiagoDiagoDiago,
  PointsSectionVFFF_DiagoDiagoSpars,
  PointsSectionVFFF_DiagoDiagoSpaco,
  PointsSectionVFFF_SparsDiagoDense,
  PointsSectionVFFF_SparsDiagoDiago,
  PointsSectionVFFF_SparsDiagoSpars,
  PointsSectionVFFF_SparsDiagoSpaco,
  PointsSectionVFFF_SpacoDiagoDense,
  PointsSectionVFFF_SpacoDiagoDiago,
  PointsSectionVFFF_SpacoDiagoSpars,
  PointsSectionVFFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVFFD_DenseDiagoDense,
  PointsSectionVFFD_DenseDiagoDiago,
  PointsSectionVFFD_DenseDiagoSpars,
  PointsSectionVFFD_DenseDiagoSpaco,
  PointsSectionVFFD_DiagoDiagoDense,
  PointsSectionVFFD_DiagoDiagoDiago,
  PointsSectionVFFD_DiagoDiagoSpars,
  PointsSectionVFFD_DiagoDiagoSpaco,
  PointsSectionVFFD_SparsDiagoDense,
  PointsSectionVFFD_SparsDiagoDiago,
  PointsSectionVFFD_SparsDiagoSpars,
  PointsSectionVFFD_SparsDiagoSpaco,
  PointsSectionVFFD_SpacoDiagoDense,
  PointsSectionVFFD_SpacoDiagoDiago,
  PointsSectionVFFD_SpacoDiagoSpars,
  PointsSectionVFFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVFDF_DenseDiagoDense,
  PointsSectionVFDF_DenseDiagoDiago,
  PointsSectionVFDF_DenseDiagoSpars,
  PointsSectionVFDF_DenseDiagoSpaco,
  PointsSectionVFDF_DiagoDiagoDense,
  PointsSectionVFDF_DiagoDiagoDiago,
  PointsSectionVFDF_DiagoDiagoSpars,
  PointsSectionVFDF_DiagoDiagoSpaco,
  PointsSectionVFDF_SparsDiagoDense,
  PointsSectionVFDF_SparsDiagoDiago,
  PointsSectionVFDF_SparsDiagoSpars,
  PointsSectionVFDF_SparsDiagoSpaco,
  PointsSectionVFDF_SpacoDiagoDense,
  PointsSectionVFDF_SpacoDiagoDiago,
  PointsSectionVFDF_SpacoDiagoSpars,
  PointsSectionVFDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVFDD_DenseDiagoDense,
  PointsSectionVFDD_DenseDiagoDiago,
  PointsSectionVFDD_DenseDiagoSpars,
  PointsSectionVFDD_DenseDiagoSpaco,
  PointsSectionVFDD_DiagoDiagoDense,
  PointsSectionVFDD_DiagoDiagoDiago,
  PointsSectionVFDD_DiagoDiagoSpars,
  PointsSectionVFDD_DiagoDiagoSpaco,
  PointsSectionVFDD_SparsDiagoDense,
  PointsSectionVFDD_SparsDiagoDiago,
  PointsSectionVFDD_SparsDiagoSpars,
  PointsSectionVFDD_SparsDiagoSpaco,
  PointsSectionVFDD_SpacoDiagoDense,
  PointsSectionVFDD_SpacoDiagoDiago,
  PointsSectionVFDD_SpacoDiagoSpars,
  PointsSectionVFDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVDFF_DenseDiagoDense,
  PointsSectionVDFF_DenseDiagoDiago,
  PointsSectionVDFF_DenseDiagoSpars,
  PointsSectionVDFF_DenseDiagoSpaco,
  PointsSectionVDFF_DiagoDiagoDense,
  PointsSectionVDFF_DiagoDiagoDiago,
  PointsSectionVDFF_DiagoDiagoSpars,
  PointsSectionVDFF_DiagoDiagoSpaco,
  PointsSectionVDFF_SparsDiagoDense,
  PointsSectionVDFF_SparsDiagoDiago,
  PointsSectionVDFF_SparsDiagoSpars,
  PointsSectionVDFF_SparsDiagoSpaco,
  PointsSectionVDFF_SpacoDiagoDense,
  PointsSectionVDFF_SpacoDiagoDiago,
  PointsSectionVDFF_SpacoDiagoSpars,
  PointsSectionVDFF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVDFD_DenseDiagoDense,
  PointsSectionVDFD_DenseDiagoDiago,
  PointsSectionVDFD_DenseDiagoSpars,
  PointsSectionVDFD_DenseDiagoSpaco,
  PointsSectionVDFD_DiagoDiagoDense,
  PointsSectionVDFD_DiagoDiagoDiago,
  PointsSectionVDFD_DiagoDiagoSpars,
  PointsSectionVDFD_DiagoDiagoSpaco,
  PointsSectionVDFD_SparsDiagoDense,
  PointsSectionVDFD_SparsDiagoDiago,
  PointsSectionVDFD_SparsDiagoSpars,
  PointsSectionVDFD_SparsDiagoSpaco,
  PointsSectionVDFD_SpacoDiagoDense,
  PointsSectionVDFD_SpacoDiagoDiago,
  PointsSectionVDFD_SpacoDiagoSpars,
  PointsSectionVDFD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVDDF_DenseDiagoDense,
  PointsSectionVDDF_DenseDiagoDiago,
  PointsSectionVDDF_DenseDiagoSpars,
  PointsSectionVDDF_DenseDiagoSpaco,
  PointsSectionVDDF_DiagoDiagoDense,
  PointsSectionVDDF_DiagoDiagoDiago,
  PointsSectionVDDF_DiagoDiagoSpars,
  PointsSectionVDDF_DiagoDiagoSpaco,
  PointsSectionVDDF_SparsDiagoDense,
  PointsSectionVDDF_SparsDiagoDiago,
  PointsSectionVDDF_SparsDiagoSpars,
  PointsSectionVDDF_SparsDiagoSpaco,
  PointsSectionVDDF_SpacoDiagoDense,
  PointsSectionVDDF_SpacoDiagoDiago,
  PointsSectionVDDF_SpacoDiagoSpars,
  PointsSectionVDDF_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVDDD_DenseDiagoDense,
  PointsSectionVDDD_DenseDiagoDiago,
  PointsSectionVDDD_DenseDiagoSpars,
  PointsSectionVDDD_DenseDiagoSpaco,
  PointsSectionVDDD_DiagoDiagoDense,
  PointsSectionVDDD_DiagoDiagoDiago,
  PointsSectionVDDD_DiagoDiagoSpars,
  PointsSectionVDDD_DiagoDiagoSpaco,
  PointsSectionVDDD_SparsDiagoDense,
  PointsSectionVDDD_SparsDiagoDiago,
  PointsSectionVDDD_SparsDiagoSpars,
  PointsSectionVDDD_SparsDiagoSpaco,
  PointsSectionVDDD_SpacoDiagoDense,
  PointsSectionVDDD_SpacoDiagoDiago,
  PointsSectionVDDD_SpacoDiagoSpars,
  PointsSectionVDDD_SpacoDiagoSpaco,
  //----------------------------------
  PointsSectionVVFF_DenseDiagoDense,
  PointsSectionVVFF_DenseDiagoDiago,
  PointsSectionVVFF_DenseDiagoSpars,
  PointsSectionVVFF_DenseDiagoSpaco,
  PointsSectionVVFF_DiagoDiagoDense,
  PointsSectionVVFF_DiagoDiagoDiago,
  PointsSectionVVFF_DiagoDiagoSpars,
  PointsSectionVVFF_DiagoDiagoSpaco,
  PointsSectionVVFF_SparsDiagoDense,
  PointsSectionVVFF_SparsDiagoDiago,
  PointsSectionVVFF_SparsDiagoSpars,
  PointsSectionVVFF_SparsDiagoSpaco,
  PointsSectionVVFF_SpacoDiagoDense,
  PointsSectionVVFF_SpacoDiagoDiago,
  PointsSectionVVFF_SpacoDiagoSpars,
  PointsSectionVVFF_SpacoDiagoSpaco,
  //----------------------------------
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
  PointsSectionVVFD_SpacoDiagoSpaco,
  //----------------------------------
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
  PointsSectionVVDF_SpacoDiagoSpaco,
  //----------------------------------
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