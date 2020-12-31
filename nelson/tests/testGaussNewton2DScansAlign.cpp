#define _USE_MATH_DEFINES
#include <cmath>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/SingleSection.hpp"
#include "nelson/GaussNewton.hpp"
#include "nelson/EdgeBinary.hpp"


#include "lie/SE2.h"
#include "lie/SE2Alg.h"
#include "lie/SE2Exp.h"
#include "lie/SE2Params.h"

#include <array>
#include <vector>

#include <iostream>


namespace constants {
  constexpr int poseParSize = 3;
#ifdef NDEBUG
  constexpr int pointsPerScan = 30000;
  constexpr int numPoses = 20;
#else
  constexpr int pointsPerScan = 30;
  constexpr int numPoses = 10;
#endif
};


template<class Section>
class PointLineEdge : public nelson::EdgeBinarySingleSectionCRPT<Section, PointLineEdge<Section>> {
  // inputs, local coordinates
  Eigen::Matrix2Xd model_points;
  Eigen::Matrix2Xd model_normals;
  Eigen::Matrix2Xd target_points;

  // world coordinates
  Eigen::Matrix2Xd model_points_w;
  Eigen::Matrix2Xd model_normals_w;
  Eigen::Matrix2Xd target_points_w;
  Eigen::RowVectorXd tx_ny_m_ty_nx;
  Eigen::RowVectorXd errors_ppl;

  // temporary H and b
  Eigen::Matrix3d Hs;
  Eigen::Vector3d bs;

public:
  PointLineEdge(
    const Eigen::Matrix2Xd& model_points,
    const Eigen::Matrix2Xd& model_normals,
    const Eigen::Matrix2Xd& target_points
  ) :
    model_points(model_points),
    model_normals(model_normals),
    target_points(target_points),
    model_points_w(2, model_points.cols()),
    model_normals_w(2, model_points.cols()),
    target_points_w(2, model_points.cols()),
    tx_ny_m_ty_nx(model_points.cols()),
    errors_ppl(model_points.cols())
  {

  }

  void update(bool hessians) override {
    const auto& modelPose = this->parameter_1();
    const auto& targetPose = this->parameter_2();

    model_points_w = modelPose.pose * model_points;
    model_normals_w = modelPose.pose.linear() * model_normals;
    target_points_w = targetPose.pose * target_points;
    errors_ppl = (model_points_w - target_points_w).cwiseProduct(model_normals_w).colwise().sum();

    this->setChi2(errors_ppl.squaredNorm());

    if (hessians) {
      tx_ny_m_ty_nx = target_points_w.row(0).cwiseProduct(model_normals_w.row(1)) - target_points_w.row(1).cwiseProduct(model_normals_w.row(0));

      Eigen::Vector2d N2 = model_normals_w.rowwise().squaredNorm();
      double NxNy = model_normals_w.row(0).cwiseProduct(model_normals_w.row(1)).sum();
      double NxC = model_normals_w.row(0).cwiseProduct(tx_ny_m_ty_nx).sum();
      double NyC = model_normals_w.row(1).cwiseProduct(tx_ny_m_ty_nx).sum();
      double C2 = tx_ny_m_ty_nx.squaredNorm();

      Hs <<
        N2(0), NxNy, NxC,
        NxNy, N2(1), NyC,
        NxC, NyC, C2;

      bs(0) = model_normals_w.row(0).cwiseProduct(errors_ppl).sum();
      bs(1) = model_normals_w.row(1).cwiseProduct(errors_ppl).sum();
      bs(2) = tx_ny_m_ty_nx.cwiseProduct(errors_ppl).sum();

    }

  }

  template<class Derived1, class Derived2>
  void updateH11Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    H.noalias() += Hs;
    b.noalias() += bs;
  }
  template<class Derived>
  void updateH12Block(Eigen::MatrixBase<Derived>& H) {
    H.noalias() -= Hs;
  }
  template<class Derived1, class Derived2>
  void updateH22Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    H.noalias() += Hs;
    b.noalias() -= bs;
  }

};

class SE2Pose {
public:
  lie::Pose2Dd pose;

  template<typename Derived>
  void oplus(const Eigen::MatrixBase<Derived>& oplus) {
    pose = lie::params2Pose(lie::pose2Params(lie::exp(lie::SE2Algd(lie::SE2Alg<double>(oplus))) * pose));
  }

};

template<int matType>
class SE2PoseSectionFF : public nelson::SingleSection<SE2PoseSectionFF<matType>, SE2Pose, matType, double, constants::poseParSize, constants::numPoses> {
  std::array<SE2Pose, constants::numPoses> _poses;
  SE2Pose _fixedPose;
public:
  SE2PoseSectionFF() {
    this->parametersReady();
  }

  virtual const SE2Pose& parameter(nelson::NodeId i) const {
    if (i.isVariable()) return _poses[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPose;
    }
  }
  virtual SE2Pose& parameter(nelson::NodeId i) {
    if (i.isVariable()) return _poses[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPose;
    }
  }

  int numFixedParameters() const override {
    return 1;
  }

  void oplus(const typename Hessian::VecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(i));
    }
  }

};

using SE2PoseSectionFF_BlockDense = SE2PoseSectionFF<mat::BlockDense>;
using SE2PoseSectionFF_BlockDiagonal = SE2PoseSectionFF<mat::BlockDiagonal>;
using SE2PoseSectionFF_BlockSparse = SE2PoseSectionFF<mat::BlockSparse>;
using SE2PoseSectionFF_BlockCoeffSparse = SE2PoseSectionFF<mat::BlockCoeffSparse>;



TEMPLATE_TEST_CASE("GaussNewton", "[GaussNewton]",
  SE2PoseSectionFF_BlockDense, SE2PoseSectionFF_BlockDiagonal, SE2PoseSectionFF_BlockSparse, SE2PoseSectionFF_BlockCoeffSparse
)
{
  std::cout << "-------------------------------------------------------" << std::endl;

  //SE2PoseSectionFF_BlockDense optProblem; assert(false); // change me to TestType
  TestType optProblem;
  //  REQUIRE(pss.parameterSize() == secSizeFix);
  REQUIRE(optProblem.numParameters() == constants::numPoses);

  std::cout << "Create World Points ..." << std::endl;
  Eigen::Matrix2Xd worldPoints = Eigen::Vector2d(100, 100).asDiagonal() * Eigen::Matrix2Xd::Random(2, constants::pointsPerScan);
  std::cout << " ... done!" << std::endl;

  std::cout << "Create World Normals ..." << std::endl;
  Eigen::Matrix2Xd worldNormals = Eigen::Matrix2Xd::Zero(2, constants::pointsPerScan);
  for (int i = 0; i < constants::pointsPerScan; i++) {
    do {
      worldNormals.col(i).setRandom();
    } while (worldNormals.col(i).squaredNorm() < 1e-3);
    worldNormals.col(i).normalize();
  }
  std::cout << " ... done!" << std::endl;

  std::cout << "Create Local Scans ..." << std::endl;
  std::vector<lie::Pose2Dd> scanPoses(constants::numPoses + 1);
  std::vector<Eigen::Matrix2Xd>  scans(constants::numPoses + 1);
  std::vector<Eigen::Matrix2Xd>  normals(constants::numPoses + 1);
  for (int i = 0; i < constants::numPoses + 1; i++) {
    scanPoses[i] = lie::exp(lie::SE2Algd(Eigen::Vector3d(10, 10, M_PI).asDiagonal() * Eigen::Vector3d::Random()));
    scans[i] = scanPoses[i].inverse() * worldPoints; // +Eigen::Vector2d(0.1, 0.1).asDiagonal() * Eigen::Matrix2Xd::Random(2, nPoints);
    normals[i] = scanPoses[i].linear().transpose() * worldNormals;
  }
  std::cout << " ... done!" << std::endl;


  Eigen::Vector3d noiseSigma;
  SECTION("NO NOISE") {
    noiseSigma.setZero();
  }
  SECTION("WITH NOISE") {
    noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
  }

  optProblem.parameter(nelson::NodeId::fixed(0)).pose = scanPoses[0];
  for (int i = 1; i < scanPoses.size(); i++) {
    optProblem.parameter(i - 1).pose = lie::exp(lie::SE2Algd(noiseSigma.asDiagonal() * Eigen::Vector3d::Random())) * scanPoses[i];
  }

  // add edges
  if (optProblem.matType() != mat::BlockDiagonal) {
    for (int i = 0; i < scanPoses.size(); i++) {
      for (int j = i + 1; j < scanPoses.size(); j++) {
        if (i == 0) {
          optProblem.addEdge(nelson::NodeId::fixed(0), j - 1, new PointLineEdge<TestType>(scans[i], normals[i], scans[j]));
        }
        else {
          optProblem.addEdge(i - 1, j - 1, new PointLineEdge<TestType>(scans[i], normals[i], scans[j]));
        }
      }
    }
  }
  else {
    for (int i = 1; i < scanPoses.size(); i++) {
      optProblem.addEdge(nelson::NodeId::fixed(0), i - 1, new PointLineEdge<TestType>(scans[0], normals[0], scans[i]));
    }
  }

  optProblem.structureReady();

  optProblem.update(true);
  std::cout << "chi2 BEFORE " << optProblem.hessian().chi2() << std::endl;

  // typename nelson::SolverCholeskyDense< TestType::Hessian::Traits::matType, typename TestType::Hessian::Traits::Type, TestType::Hessian::Traits::B, TestType::Hessian::Traits::NB>::DenseWrapperT wrap;
  // wrap.set(&optProblem.hessian().H());
  // 
  // std::cout << "H MATRIX BEFORE " << std::endl <<
  //   wrap.mat() << std::endl << std::endl;

  //std::cout << "H MATRIX BEFORE (native)" << std::endl <<
  //  optProblem.hessian().H().mat().coeffs() << std::endl << std::endl;

  nelson::GaussNewton<TestType::Hessian::Traits::matType, typename TestType::Hessian::Traits::Type, TestType::Hessian::Traits::B, TestType::Hessian::Traits::NB>  gn;
  auto tc = gn.solve(optProblem);
  std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;

  optProblem.update(true);
  std::cout << "chi2 AFTER " << optProblem.hessian().chi2() << std::endl;

  REQUIRE(optProblem.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());

}
