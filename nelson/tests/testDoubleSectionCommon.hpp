#pragma once
#include "testDoubleSectionCommon.h"

#include "nelson/GaussNewton.hpp"
#include "nelson/LevenbergMarquardt.hpp"

template<class TestType>
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
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskyDense, nelson::matrixWrapperDense, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskyDense, nelson::matrixWrapperDense, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskyDense, nelson::matrixWrapperSparse, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskyDense, nelson::matrixWrapperSparse, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskySparse, nelson::matrixWrapperDense, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskySparse, nelson::matrixWrapperDense, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskySparse, nelson::matrixWrapperSparse, nelson::matrixWrapperDense, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
  {
    nelson::GaussNewton <typename nelson::SolverTraits<nelson::solverCholeskySchur>::Solver<typename TestType::Hessian::Traits, nelson::solverCholeskySparse, nelson::matrixWrapperSparse, nelson::matrixWrapperSparse, nelson::choleskyAMDOrdering, nelson::choleskyAMDOrdering> > gn;
    pss.addNoise(0.5);
    auto tc = gn.solve(pss);
    std::cout << nelson::GaussNewtonUtils::toString(tc) << std::endl;
    std::cout << "stats " << gn.stats().toString() << std::endl;
    REQUIRE(pss.hessian().chi2() < Eigen::NumTraits<double>::dummy_precision());
  }
}
