#pragma once
#include "GaussNewton.h"

#include "SolverCholeskyDense.hpp"
#include "SolverCholeskySparse.hpp"
#include "SolverCholeskySchur.hpp"

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

    // compute initial error
    op.update(true);
    _stats.addIteration(iter, op.hessian().chi2());

    // check termination criterion on bVector
    if (op.hessian().maxAbsValBVect() <= _settings.epsBVector) {
      return GaussNewtonTerminationReason::SuccessAtStart;
    }

    // create the solver
    _solver.init(op.hessian().H(), op.hessian().b());

    iter = 0;
    while (iter < _settings.maxNumIt) {

      bool solveOk = _solver.computeIncrement(op.hessian().H(), op.hessian().b(), _settings.relLambda, _settings.absLambda);
      // check solution validity
      if (!solveOk) {
        return GaussNewtonTerminationReason::SolverFailure;
      }

      // evaluate if the increment is very small and stop in case
      if (iter >= _settings.minNumIt && _solver.incrementVectorSquaredNorm() <= _settings.epsIncSquare) {
        return GaussNewtonTerminationReason::SuccessSmallIncrement;
      }

      // apply the increment
      op.oplus(_solver.incrementVector());


      // compute error and jacobian with this solution, prepare for next iteration
      op.update(true);
      _stats.addIteration(iter, op.hessian().chi2());

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


