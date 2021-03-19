#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/GaussNewton.hpp"
#include "nelson/LevenbergMarquardt.hpp"

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
  void updateH12Block(Eigen::MatrixBase<Derived1>& H, bool transpose) {
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
  void updateH12Block(Eigen::MatrixBase<Derived1>& H, bool transpose) {
    assert(transpose == false);
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
  void updateH12Block(Eigen::MatrixBase<Derived1>& H, bool transpose) {
    assert(transpose == false);
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
      _points2d[i].p2d += inc.bU().segment(this->user2internalIndexesU()(i));
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

  SECTION("permute") {
    if (pss.matTypeW() != mat::BlockDiagonal) {
      std::cout << "************** PERMUTE ***************" << std::endl;
      Eigen::VectorXi order = pss.user2internalIndexesU();
      std::random_shuffle(order.data(), order.data() + order.size());
      pss.setUser2InternalIndexesU(order);
    }
    else {
      std::cout << "### W DIAGONAL, skip permute ###" << std::endl;
    }
  }
  SECTION("amd-permute") {
    if (pss.matTypeW() != mat::BlockDiagonal) {
      std::cout << "************** PERMUTE AMD ***************" << std::endl;
      pss.permuteAMD_SchurU();
    }
    else {
      std::cout << "### W DIAGONAL, skip permute ###" << std::endl;
    }
  }
  SECTION("standard") {
    std::cout << "************** STANDARD ***************" << std::endl;
  }


  pss.structureReady();



  pss.update(true);

  {
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverseWWtMult>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering> >;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }

  {
    using SolverAlgorithm = Solver <typename nelson::SolverTraits<nelson::solverCholeskySchurDiagBlockInverseWWtMult>::Solver<typename TestType::Hessian::Traits, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering > > ;
    SolverAlgorithm gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << SolverAlgorithm::Utils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }


}

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

TEMPLATE_TEST_CASE("DoubleSection-FFDD", "[DoubleSection-FFDD]",
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
  PointsSectionFFDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}