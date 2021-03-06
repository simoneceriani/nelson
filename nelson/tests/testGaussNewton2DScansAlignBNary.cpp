#define _USE_MATH_DEFINES
#include <cmath>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/SingleSection.hpp"
#include "nelson/GaussNewton.hpp"
#include "nelson/LevenbergMarquardt.hpp"
#include "nelson/EdgeNary.hpp"


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
  constexpr int pointsPerScan = 3000;
  constexpr int numPoses = 15;
#else
  constexpr int pointsPerScan = 30;
  constexpr int numPoses = 10;
#endif
};


template<class Section>
class PointLineEdge : public nelson::EdgeBNarySectionBaseCRPT<Section, 1, 1, typename Section::EdgeBinaryAdapter, PointLineEdge<Section>> {
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
    const auto& modelPose = this->parameter_1(0);
    const auto& targetPose = this->parameter_2(0);

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
  void updateHUBlock(int i, Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    assert(i == 0);
    H.noalias() += Hs;
    b.noalias() += bs;
  }
  template<class Derived>
  void updateHUBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    assert(false);
  }
  template<class Derived1, class Derived2>
  void updateHVBlock(int i, Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    assert(i == 0);
    H.noalias() += Hs;
    b.noalias() -= bs;
  }
  template<class Derived>
  void updateHVBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    assert(false);
  }
  template<class Derived>
  void updateHWBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    assert(i == 0 && j == 0);
    Hij.noalias() -= Hs;
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

//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionFF : public nelson::SingleSection<SE2PoseSectionFF<matTypeV>, SE2Pose, matTypeV, double, constants::poseParSize, constants::numPoses> {
  std::array<SE2Pose, constants::numPoses> _poses;
  std::array<SE2Pose, constants::numPoses> _bckposes;
  SE2Pose _fixedPose;
  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionFF<matTypeV>, SE2Pose, matTypeV, double, constants::poseParSize, constants::numPoses>;
public:
  SE2PoseSectionFF() {
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionFF_BlockDense = SE2PoseSectionFF<mat::BlockDense>;
using SE2PoseSectionFF_BlockDiagonal = SE2PoseSectionFF<mat::BlockDiagonal>;
using SE2PoseSectionFF_BlockSparse = SE2PoseSectionFF<mat::BlockSparse>;
using SE2PoseSectionFF_BlockCoeffSparse = SE2PoseSectionFF<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionFD : public nelson::SingleSection<SE2PoseSectionFD<matTypeV>, SE2Pose, matTypeV, double, constants::poseParSize, mat::Dynamic> {
  std::vector<SE2Pose> _poses;
  std::vector<SE2Pose> _bckposes;
  SE2Pose _fixedPose;
  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionFD<matTypeV>, SE2Pose, matTypeV, double, constants::poseParSize, mat::Dynamic>;
public:
  SE2PoseSectionFD() {
    _poses.resize(constants::numPoses);
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  int numParameters() const override {
    return _poses.size();
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionFD_BlockDense = SE2PoseSectionFD<mat::BlockDense>;
using SE2PoseSectionFD_BlockDiagonal = SE2PoseSectionFD<mat::BlockDiagonal>;
using SE2PoseSectionFD_BlockSparse = SE2PoseSectionFD<mat::BlockSparse>;
using SE2PoseSectionFD_BlockCoeffSparse = SE2PoseSectionFD<mat::BlockCoeffSparse>;
//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionDF : public nelson::SingleSection<SE2PoseSectionDF<matTypeV>, SE2Pose, matTypeV, double, mat::Dynamic, constants::numPoses> {
  std::array<SE2Pose, constants::numPoses> _poses;
  std::array<SE2Pose, constants::numPoses> _bckposes;
  SE2Pose _fixedPose;

  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionDF<matTypeV>, SE2Pose, matTypeV, double, mat::Dynamic, constants::numPoses>;
public:
  SE2PoseSectionDF() {
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  int parameterSize() const override {
    return constants::poseParSize;
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionDF_BlockDense = SE2PoseSectionDF<mat::BlockDense>;
using SE2PoseSectionDF_BlockDiagonal = SE2PoseSectionDF<mat::BlockDiagonal>;
using SE2PoseSectionDF_BlockSparse = SE2PoseSectionDF<mat::BlockSparse>;
using SE2PoseSectionDF_BlockCoeffSparse = SE2PoseSectionDF<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionDD : public nelson::SingleSection<SE2PoseSectionDD<matTypeV>, SE2Pose, matTypeV, double, mat::Dynamic, mat::Dynamic> {
  std::vector<SE2Pose> _poses;
  std::vector<SE2Pose> _bckposes;
  SE2Pose _fixedPose;

  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionDD<matTypeV>, SE2Pose, matTypeV, double, mat::Dynamic, mat::Dynamic>;
public:
  SE2PoseSectionDD() {
    _poses.resize(constants::numPoses);
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  int numParameters() const override {
    return _poses.size();
  }

  int parameterSize() const override {
    return constants::poseParSize;
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionDD_BlockDense = SE2PoseSectionDD<mat::BlockDense>;
using SE2PoseSectionDD_BlockDiagonal = SE2PoseSectionDD<mat::BlockDiagonal>;
using SE2PoseSectionDD_BlockSparse = SE2PoseSectionDD<mat::BlockSparse>;
using SE2PoseSectionDD_BlockCoeffSparse = SE2PoseSectionDD<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionVF : public nelson::SingleSection<SE2PoseSectionVF<matTypeV>, SE2Pose, matTypeV, double, mat::Variable, constants::numPoses> {
  std::array<SE2Pose, constants::numPoses> _poses;
  std::array<SE2Pose, constants::numPoses> _bckposes;
  std::vector<int> _sizes;
  SE2Pose _fixedPose;

  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionVF<matTypeV>, SE2Pose, matTypeV, double, mat::Variable, constants::numPoses>;
public:
  SE2PoseSectionVF() : _sizes(constants::numPoses, constants::poseParSize) {
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  const std::vector<int>& parameterSize() const override {
    return _sizes;
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionVF_BlockDense = SE2PoseSectionVF<mat::BlockDense>;
using SE2PoseSectionVF_BlockDiagonal = SE2PoseSectionVF<mat::BlockDiagonal>;
using SE2PoseSectionVF_BlockSparse = SE2PoseSectionVF<mat::BlockSparse>;
using SE2PoseSectionVF_BlockCoeffSparse = SE2PoseSectionVF<mat::BlockCoeffSparse>;
//-----------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------

template<int matTypeV>
class SE2PoseSectionVD : public nelson::SingleSection<SE2PoseSectionVD<matTypeV>, SE2Pose, matTypeV, double, mat::Variable, mat::Dynamic> {
  std::vector<SE2Pose> _poses;
  std::vector<SE2Pose> _bckposes;
  std::vector<int> _sizes;
  SE2Pose _fixedPose;

  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionVD<matTypeV>, SE2Pose, matTypeV, double, mat::Variable, mat::Dynamic>;
public:
  SE2PoseSectionVD() : _sizes(constants::numPoses, constants::poseParSize) {
    _poses.resize(constants::numPoses);
    this->parametersReady();
  }

  void backupSolution() {
    _bckposes = _poses;
  }

  void rollbackSolution() {
    _poses = _bckposes;
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

  const std::vector<int>& parameterSize() const override {
    return _sizes;
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(this->user2internalIndexes()(i)));
    }
  }

};

using SE2PoseSectionVD_BlockDense = SE2PoseSectionVD<mat::BlockDense>;
using SE2PoseSectionVD_BlockDiagonal = SE2PoseSectionVD<mat::BlockDiagonal>;
using SE2PoseSectionVD_BlockSparse = SE2PoseSectionVD<mat::BlockSparse>;
using SE2PoseSectionVD_BlockCoeffSparse = SE2PoseSectionVD<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE_SIG("GaussNewton-LevenbergMarquardt", "[GaussNewton-LevenbergMarquardt]", ((class ProblemType, int solverType), ProblemType, solverType),
  (SE2PoseSectionFF_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionFF_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionFF_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionFF_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionFD_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionFD_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionFD_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionFD_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionDF_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionDF_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionDF_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionDF_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionDD_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionDD_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionDD_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionDD_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionVF_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionVF_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionVF_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionVF_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionVD_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionVD_BlockDiagonal, nelson::solverCholeskyDense), (SE2PoseSectionVD_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionVD_BlockCoeffSparse, nelson::solverCholeskyDense),
  //---------------------------------------------------------------------------------------------------------------------
  (SE2PoseSectionFF_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionFF_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionFF_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionFF_BlockCoeffSparse, nelson::solverCholeskySparse),
  (SE2PoseSectionFD_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionFD_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionFD_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionFD_BlockCoeffSparse, nelson::solverCholeskySparse),
  (SE2PoseSectionDF_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionDF_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionDF_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionDF_BlockCoeffSparse, nelson::solverCholeskySparse),
  (SE2PoseSectionDD_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionDD_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionDD_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionDD_BlockCoeffSparse, nelson::solverCholeskySparse),
  (SE2PoseSectionVF_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionVF_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionVF_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionVF_BlockCoeffSparse, nelson::solverCholeskySparse),
  (SE2PoseSectionVD_BlockDense, nelson::solverCholeskySparse), (SE2PoseSectionVD_BlockDiagonal, nelson::solverCholeskySparse), (SE2PoseSectionVD_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionVD_BlockCoeffSparse, nelson::solverCholeskySparse)
)
{
  for (int repeat = 0; repeat < 6; repeat++) {
    SECTION("repeat test for time " + std::to_string(repeat)) {
      std::cout << " *********** repeat " << repeat << " *************" << std::endl;
      auto startTime = std::chrono::steady_clock::now();
      //SE2PoseSectionFF_BlockDense optProblem; assert(false); // change me to ProblemType
      ProblemType optProblem;

      if (repeat == 1 || repeat == 3 || repeat == 5) {
        optProblem.settings().hessianUpdateParallelSettings.setNumThreadsMax();
        optProblem.settings().edgeEvalParallelSettings.setNumThreadsMax();
      }

      int ordering = 0;
      if (repeat == 0 || repeat == 1) {
        ordering = 1; //AMD
      }
      else if (repeat == 2 || repeat == 3) {
        ordering = 2; //SHUFFLE
      }


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
      bool fullEdges;
      bool changeDiagInSolver;
      bool solveGN = false;

      //---------------------- GN
      SECTION("NO NOISE - FULL - PURE GN") {
        noiseSigma.setZero();
        fullEdges = true;
        changeDiagInSolver = false;
        solveGN = true;
      }
      SECTION("WITH NOISE - FULL - PURE GN") {
        noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
        fullEdges = true;
        changeDiagInSolver = false;
        solveGN = true;
      }
      if (optProblem.matType() != mat::BlockDiagonal) {
        SECTION("NO NOISE - SPARSE - PURE GN") {
          noiseSigma.setZero();
          fullEdges = false;
          changeDiagInSolver = false;
          solveGN = true;
        }
        SECTION("WITH NOISE - SPARSE - PURE GN") {
          noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
          fullEdges = false;
          changeDiagInSolver = false;
          solveGN = true;
        }
      }
      else fullEdges = false;

      SECTION("NO NOISE - FULL - DAMPED GN") {
        noiseSigma.setZero();
        fullEdges = true;
        changeDiagInSolver = true;
        solveGN = true;
      }
      SECTION("WITH NOISE - FULL - DAMPED GN") {
        noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
        fullEdges = true;
        changeDiagInSolver = true;
        solveGN = true;
      }
      if (optProblem.matType() != mat::BlockDiagonal) {
        SECTION("NO NOISE - SPARSE - DAMPED GN") {
          noiseSigma.setZero();
          fullEdges = false;
          changeDiagInSolver = true;
          solveGN = true;
        }
        SECTION("WITH NOISE - SPARSE - DAMPED GN") {
          noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
          fullEdges = false;
          changeDiagInSolver = true;
          solveGN = true;
        }
      }
      else fullEdges = false;
      //------------------------------- levenberg
      SECTION("NO NOISE - FULL - PURE GN") {
        noiseSigma.setZero();
        fullEdges = true;
        changeDiagInSolver = false;
        solveGN = false;
      }
      SECTION("WITH NOISE - FULL - PURE GN") {
        noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
        fullEdges = true;
        changeDiagInSolver = false;
        solveGN = false;
      }
      if (optProblem.matType() != mat::BlockDiagonal) {
        SECTION("NO NOISE - SPARSE - PURE GN") {
          noiseSigma.setZero();
          fullEdges = false;
          changeDiagInSolver = false;
          solveGN = false;
        }
        SECTION("WITH NOISE - SPARSE - PURE GN") {
          noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
          fullEdges = false;
          changeDiagInSolver = false;
          solveGN = false;
        }
      }
      else fullEdges = false;


      optProblem.parameter(nelson::NodeId::fixed(0)).pose = scanPoses[0];
      for (int i = 1; i < scanPoses.size(); i++) {
        optProblem.parameter(i - 1).pose = lie::exp(lie::SE2Algd(noiseSigma.asDiagonal() * Eigen::Vector3d::Random())) * scanPoses[i];
      }
     
      // add edges
      if (optProblem.matType() != mat::BlockDiagonal) {
        int nEdgesFull = (scanPoses.size() * scanPoses.size() - scanPoses.size()) / 2;
        optProblem.reserveEdges(nEdgesFull);

        if (fullEdges) {
          for (int i = 0; i < scanPoses.size(); i++) {
            for (int j = i + 1; j < scanPoses.size(); j++) {
              if (i == 0) {
                optProblem.addEdge({ nelson::NodeId::fixed(0)},{ j - 1 }, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
              else {
                optProblem.addEdge({ i - 1},{ j - 1 }, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
            }
          }
        }
        else {

          for (int i = 1; i < scanPoses.size(); i++) {
            optProblem.addEdge({ nelson::NodeId::fixed(0)},{ i - 1 }, new PointLineEdge<ProblemType>(scans[0], normals[0], scans[i]));
          }

          for (int i = 1; i < scanPoses.size() - 1; i++) {
            int nj = rand() % (scanPoses.size() - i - 1);
            for (int cj = 0; cj < nj; cj++) {
              int j = rand() % (scanPoses.size() - i - 1);
              j += i + 1;
              if (i == 0) {
                optProblem.addEdge({ nelson::NodeId::fixed(0)},{ j - 1 }, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
              else {
                optProblem.addEdge({i - 1},{ j - 1}, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
            }
          }

        }
      }
      else {
        optProblem.reserveEdges(scanPoses.size());
        for (int i = 1; i < scanPoses.size(); i++) {
          optProblem.addEdge({ nelson::NodeId::fixed(0)},{ i - 1 }, new PointLineEdge<ProblemType>(scans[0], normals[0], scans[i]));
        }
      }

      
      if (ordering == 1) {
        optProblem.permuteAMD();
      }
      else if (ordering == 2) {
        Eigen::VectorXi order = optProblem.user2internalIndexes();
        std::random_shuffle(order.data(), order.data() + order.size());
        optProblem.setUser2InternalIndexes(order);
      }

      optProblem.structureReady();

      optProblem.update(true);

      if (!solveGN) {
        nelson::LevenbergMarquardt<typename nelson::SolverTraits<solverType>::template Solver<typename ProblemType::Hessian::Traits, nelson::choleskyAMDOrdering> > lm;
        auto tc = lm.solve(optProblem);
        std::cout << nelson::LevenbergMarquardtUtils::toString(tc) << std::endl;
        std::cout << "stats " << lm.stats().toString() << std::endl;
      }
      else {
        nelson::GaussNewton <typename nelson::SolverTraits<solverType>::template Solver<typename ProblemType::Hessian::Traits, nelson::choleskyAMDOrdering> > gn;
        if (changeDiagInSolver) {
          gn.settings().relLambda = 0.01;
          gn.settings().absLambda = 0;
        }
        auto tc = gn.solve(optProblem);
        std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
        std::cout << "stats " << gn.stats().toString() << std::endl;
      }

      optProblem.update(true);
      std::cout << "chi2 AFTER " << optProblem.hessian().chi2() << std::endl;

      REQUIRE(optProblem.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());

      auto endTime = std::chrono::steady_clock::now();
      std::cout << "TOTAL TIME " << std::chrono::duration<double>(endTime - startTime).count() << std::endl;

    }

  }

}
