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

#ifdef NDEBUG
static constexpr int poseParSize = 3;
static constexpr int pointsPerScan = 10000;
static const int numPoses = 32;
#else
static constexpr int poseParSize = 3;
static constexpr int pointsPerScan = 10;
static const int numPoses = 6;
#endif


template<class Section>
class PointLineEdge : public nelson::EdgeBinarySectionBaseCRPT<Section, typename Section::EdgeBinaryAdapter, PointLineEdge<Section>> {
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

template<int matTypeV>
class SE2PoseSectionFD : public nelson::SingleSection<SE2PoseSectionFD<matTypeV>, SE2Pose, matTypeV, double, poseParSize, mat::Dynamic> {
  std::vector<SE2Pose> _poses;
  SE2Pose _fixedPose;
  using SingleSectionBase = nelson::SingleSection<SE2PoseSectionFD<matTypeV>, SE2Pose, matTypeV, double, poseParSize, mat::Dynamic>;
public:
  SE2PoseSectionFD(int numPoses) {
    _poses.resize(numPoses);
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

  int numParameters() const override {
    return _poses.size();
  }

  void oplus(const typename SingleSectionBase::HessianVecType& inc) {
    for (int i = 0; i < this->numParameters(); i++) {
      _poses[i].oplus(inc.segment(i));
    }
  }

};

using SE2PoseSectionFD_BlockDense = SE2PoseSectionFD<mat::BlockDense>;
using SE2PoseSectionFD_BlockSparse = SE2PoseSectionFD<mat::BlockSparse>;
using SE2PoseSectionFD_BlockCoeffSparse = SE2PoseSectionFD<mat::BlockCoeffSparse>;
//-----------------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE_SIG("TestTiming", "[TestTiming]", ((class ProblemType, int solverType), ProblemType, solverType),
  (SE2PoseSectionFD_BlockDense, nelson::solverCholeskyDense), (SE2PoseSectionFD_BlockSparse, nelson::solverCholeskyDense), (SE2PoseSectionFD_BlockCoeffSparse, nelson::solverCholeskyDense),
  (SE2PoseSectionFD_BlockDense, nelson::solverCholeskySparse),(SE2PoseSectionFD_BlockSparse, nelson::solverCholeskySparse), (SE2PoseSectionFD_BlockCoeffSparse, nelson::solverCholeskySparse)
)
{
  for (int repeat = 0; repeat < 2; repeat++) {
    SECTION("repeat test for time " + std::to_string(repeat)) {
      std::cout << " *********** repeat " << (repeat == 0 ? "SINGLE THREAD" : " MULTI THREAD") << " *************" << std::endl;
      auto startTime = std::chrono::steady_clock::now();



      std::cout << "Create World Points ..." << std::endl;
      Eigen::Matrix2Xd worldPoints = Eigen::Vector2d(100, 100).asDiagonal() * Eigen::Matrix2Xd::Random(2, pointsPerScan);
      std::cout << " ... done!" << std::endl;

      std::cout << "Create World Normals ..." << std::endl;
      Eigen::Matrix2Xd worldNormals = Eigen::Matrix2Xd::Zero(2, pointsPerScan);
      for (int i = 0; i < pointsPerScan; i++) {
        do {
          worldNormals.col(i).setRandom();
        } while (worldNormals.col(i).squaredNorm() < 1e-3);
        worldNormals.col(i).normalize();
      }
      std::cout << " ... done!" << std::endl;

      std::cout << "Create Local Scans ..." << std::endl;
      std::vector<lie::Pose2Dd> scanPoses(numPoses + 1);
      std::vector<Eigen::Matrix2Xd>  scans(numPoses + 1);
      std::vector<Eigen::Matrix2Xd>  normals(numPoses + 1);
      for (int i = 0; i < numPoses + 1; i++) {
        scanPoses[i] = lie::exp(lie::SE2Algd(Eigen::Vector3d(10, 10, M_PI).asDiagonal() * Eigen::Vector3d::Random()));
        scans[i] = scanPoses[i].inverse() * worldPoints; // +Eigen::Vector2d(0.1, 0.1).asDiagonal() * Eigen::Matrix2Xd::Random(2, nPoints);
        normals[i] = scanPoses[i].linear().transpose() * worldNormals;
      }
      std::cout << " ... done!" << std::endl;


      Eigen::Vector3d noiseSigma;
      bool fullEdges;
      /*SECTION("WITH NOISE - FULL") {
        std::cout << "WITH NOISE - FULL" << std::endl;
        noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
        fullEdges = true;
      }*/
      SECTION("WITH NOISE - SPARSE") {
        std::cout << "WITH NOISE - SPARSE" << std::endl;
        noiseSigma = Eigen::Vector3d(0.01, 0.01, M_PI / 1000.0);
        fullEdges = false;
      }

      auto startPrepareTime = std::chrono::steady_clock::now();
      ProblemType optProblem(numPoses);
      REQUIRE(optProblem.numParameters() == numPoses);

      if (repeat == 1) {
        optProblem.settings().hessianUpdateParallelSettings.setNumThreadsMax();
        optProblem.settings().edgeEvalParallelSettings.setNumThreadsMax();
      }

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
                optProblem.addEdge(nelson::NodeId::fixed(0), j - 1, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
              else {
                optProblem.addEdge(i - 1, j - 1, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
            }
          }
          REQUIRE(optProblem.numEdges() == nEdgesFull);
        }
        else {
          for (int i = 1; i < scanPoses.size(); i++) {
            optProblem.addEdge(nelson::NodeId::fixed(0), i - 1, new PointLineEdge<ProblemType>(scans[0], normals[0], scans[i]));
          }

          for (int i = 1; i < scanPoses.size() - 1; i++) {
            int nj = rand() % (scanPoses.size() - i - 1);
            for (int cj = 0; cj < nj; cj++) {
              int j = rand() % (scanPoses.size() - i - 1);
              j += i + 1;
              if (i == 0) {
                optProblem.addEdge(nelson::NodeId::fixed(0), j - 1, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
              else {
                optProblem.addEdge(i - 1, j - 1, new PointLineEdge<ProblemType>(scans[i], normals[i], scans[j]));
              }
            }
          }

        }
      }
      else {
        // diagonal
        optProblem.reserveEdges(scanPoses.size());
        for (int i = 1; i < scanPoses.size(); i++) {
          optProblem.addEdge(nelson::NodeId::fixed(0), i - 1, new PointLineEdge<ProblemType>(scans[0], normals[0], scans[i]));
        }
      }

      optProblem.structureReady();
      auto endPrepareTime = std::chrono::steady_clock::now();
      std::cout << "PREPARE TIME " << std::chrono::duration<double>(endPrepareTime - startPrepareTime).count() << std::endl;

      optProblem.update(true);
      std::cout << "chi2 BEFORE " << optProblem.hessian().chi2() << std::endl;

      auto startSolveTime = std::chrono::steady_clock::now();
      nelson::GaussNewton <typename nelson::SolverTraits<solverType>::Solver<typename ProblemType::Hessian::Traits> > gn;
      gn.settings().minNumIt = 5;
      auto tc = gn.solve(optProblem);
      auto endSolveTime = std::chrono::steady_clock::now();

      std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
      std::cout << "SOLVE TIME " << std::chrono::duration<double>(endSolveTime - startSolveTime).count() << std::endl;
      std::cout << "num iter " << gn.numIterations() << std::endl;
      std::cout << "ITER TIME " << std::chrono::duration<double>(endSolveTime - startSolveTime).count() / gn.numIterations() << std::endl;

      optProblem.update(true);
      std::cout << "chi2 AFTER " << optProblem.hessian().chi2() << std::endl;

      REQUIRE(optProblem.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());

      auto endTime = std::chrono::steady_clock::now();
      std::cout << "TOTAL TIME " << std::chrono::duration<double>(endTime - startTime).count() << std::endl;
    }

  }

}

