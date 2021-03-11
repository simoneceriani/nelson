#pragma once
#include "GaussNewton.h"

#include "SolverCholeskyDense.hpp"
#include "SolverCholeskySparse.hpp"
#include "SolverCholeskySchur.hpp"
#include "SolverDiagonalBlocksInverseSchur.hpp"
#include "SolverDiagonalBlocksInverseWWtMultSchur.hpp"

namespace nelson {

  template<class Solver>
  GaussNewton<Solver>::GaussNewton() : iter(-1){

  }

  template<class Solver>
  GaussNewton<Solver>::GaussNewton(const GaussNewtonSettings& settings)
    : _settings(settings), iter(-1)
  {

  }

  template<class Solver>
  GaussNewton<Solver>::~GaussNewton() {

  }


  template<class Solver>
  template<class OptimizationProblem>
  GaussNewtonTerminationReason GaussNewton<Solver>::solve(OptimizationProblem& op) {

    _stats.reserve(_settings.maxNumIt);
    _timingStats.reserve(_settings.maxNumIt);

    // compute initial error
    {
      auto t0 = std::chrono::steady_clock::now();
      op.update(true);
      auto t1 = std::chrono::steady_clock::now();
      _stats.addIteration(iter, op.hessian().chi2());
      _timingStats.setFirstEvalTime(std::chrono::duration<double>(t1 - t0));
    }
    // check termination criterion on bVector
    if (op.hessian().maxAbsValBVect() <= _settings.epsBVector) {
      return GaussNewtonTerminationReason::SuccessAtStart;
    }

    // create the solver
    {
      auto t0 = std::chrono::steady_clock::now();
      _solver.init(op.hessian().H(), op.hessian().b());
      auto t1 = std::chrono::steady_clock::now();
      _timingStats.setSolverInitTime(std::chrono::duration<double>(t1 - t0));
    }

    iter = 0;
    auto tinit = std::chrono::steady_clock::now();
    while (iter < _settings.maxNumIt) {

      TimingStats::IterationTime timeStat;

      bool solveOk = false;
      {
        auto t0 = std::chrono::steady_clock::now();
        solveOk = _solver.computeIncrement(op.hessian().H(), op.hessian().b(), _settings.relLambda, _settings.absLambda);
        auto t1 = std::chrono::steady_clock::now();
        timeStat.computeIncrementTime = std::chrono::duration<double>(t1 - t0);
      }
      // check solution validity
      if (!solveOk) {
        auto tend = std::chrono::steady_clock::now();
        timeStat.overallTime = tend - tinit;
        _timingStats.addIteration(timeStat);

        return GaussNewtonTerminationReason::SolverFailure;
      }

      // evaluate if the increment is very small and stop in case
      if (iter >= _settings.minNumIt && _solver.incrementVectorSquaredNorm() <= _settings.epsIncSquare) {
        auto tend = std::chrono::steady_clock::now();
        timeStat.overallTime = tend - tinit;
        _timingStats.addIteration(timeStat);

        return GaussNewtonTerminationReason::SuccessSmallIncrement;
      }

      // apply the increment
      {
        auto t0 = std::chrono::steady_clock::now();
        op.oplus(_solver.incrementVector());
        auto t1 = std::chrono::steady_clock::now();
        timeStat.applyIncrementTime = std::chrono::duration<double>(t1 - t0);
      }

      // compute error and jacobian with this solution, prepare for next iteration
      {
        auto t0 = std::chrono::steady_clock::now();
        op.update(true);
        auto t1 = std::chrono::steady_clock::now();
        timeStat.evalTime = std::chrono::duration<double>(t1 - t0);
      }
      _stats.addIteration(iter, op.hessian().chi2());

      auto tend = std::chrono::steady_clock::now();
      timeStat.overallTime = tend - tinit;
      _timingStats.addIteration(timeStat);
      tinit = tend;

      // check termination criterion on b vector
      if (iter > _settings.minNumIt && op.hessian().maxAbsValBVect() <= _settings.epsBVector) {
        return GaussNewtonTerminationReason::SuccessEpsB;
      }

      // check termination criterion on residuals
      if (iter > _settings.minNumIt && op.hessian().chi2() < _settings.epsChi2) {
        return GaussNewtonTerminationReason::SuccessSmallResiduals;
      }

      // todo: check chi2 relative small change for success and other termination criterion

      iter++;
    }

    return GaussNewtonTerminationReason::MaxIterationReached;
  }

}


