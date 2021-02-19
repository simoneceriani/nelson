#include "nelson/DoubleSectionHessian.hpp"
#include "nelson/DoubleSection.hpp"
#include "nelson/EdgeUnary.hpp"
#include "nelson/EdgeBinary.hpp"

#include <array>
#include <iostream>

template<class TestType>
void testFunction();

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


template<class Derived, int matTypeUv, int matTypeVv, int matTypeWv, int BU, int BV, int NBU, int NBV>
class Points2d3dBase : public nelson::DoubleSection< Derived, Point2d, Point3d, matTypeUv, matTypeVv, matTypeWv, double, BU, BV, NBU, NBV> {

  std::array<Point2d, numPoints2d> _points2d;
  std::array<Point3d, numPoints3d> _points3d;
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


};