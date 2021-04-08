#pragma once
#include "LevenbergMarquardt.h"

#include "SolverCholeskyDense.hpp"
#include "SolverCholeskySparse.hpp"
#include "SolverCholeskySchur.hpp"
#include "SolverDiagonalBlocksInverseSchur.hpp"
#include "SolverDiagonalBlocksInverseWWtMultSchur.hpp"

namespace nelson {

  template<class Solver>
  LevenbergMarquardt<Solver>::LevenbergMarquardt()
    : iter(-1),
    subiter(-1)
  {

  }

  template<class Solver>
  LevenbergMarquardt<Solver>::LevenbergMarquardt(const LevenbergMarquardtSettings& settings)
    : _settings(settings),
    iter(-1),
    subiter(-1)
  {

  }

  template<class Solver>
  LevenbergMarquardt<Solver>::~LevenbergMarquardt() {

  }

  template<class Solver>
  template<class OptimizationProblem>
  LevenbergMarquardtTerminationReason LevenbergMarquardt<Solver>::solve(OptimizationProblem& op) {

    _stats.reserve(_settings.maxNumIt, _settings.maxNumSubIt);

    // compute initial error
    op.update(true);
    _stats.addIteration(iter);
    _stats.addSubIteration(subiter, op.hessian().chi2());

    // check termination criterion on bVector
    if (op.hessian().maxAbsValBVect() <= _settings.epsBVector) {
      return LevenbergMarquardtTerminationReason::SuccessAtStart;
    }

    // backup solution immediately
    op.backupSolution();
    auto oldChi2 = op.hessian().chi2();

    // create the solver
    _solver.init(op.hessian().H(), op.hessian().b());

    // init mu, V values
    auto mu = _settings.tau * _solver.maxAbsHDiag();
    auto curV = _settings.initV;


    iter = 0;
    while (iter < _settings.maxNumIt) {
      _stats.addIteration(iter);
      subiter = 0;
      bool toRepeat = false;
      do {

        // compute increment
        bool solveOk = _solver.computeIncrement(op.hessian().H(), op.hessian().b(), 0, mu);

        // evaluate if the increment is very small and stop in case
        if (solveOk && iter >= _settings.minNumIt && _solver.incrementVectorSquaredNorm() <= _settings.epsIncSquare) {
          return LevenbergMarquardtTerminationReason::SuccessSmallIncrement;
        }

        // evaluate the effects of the iteration performed
        typename Solver::Type rho = -1;
        if (solveOk) {

          // apply the increment
          op.oplus(_solver.incrementVector());

          // compute error and jacobian with this solution, prepare for next iteration
          op.update(false);
          _stats.addSubIteration(subiter, op.hessian().chi2());

          // evaluate chi2 change
          rho = op.computeRhoChi2Change(mu, _solver.incrementVector(), oldChi2);

        }
        else {
          _stats.addSubIteration(subiter, -1); //solver error
        }

        if (solveOk && rho > 0) {

          // check termination criterion on b vector
          if (iter > _settings.minNumIt && op.hessian().maxAbsValBVect() <= _settings.epsBVector) {
            return LevenbergMarquardtTerminationReason::SuccessEpsB;
          }

          // check termination criterion on residuals
          if (iter > _settings.minNumIt && op.hessian().chi2() < _settings.epsChi2) {
            return LevenbergMarquardtTerminationReason::SuccessSmallResiduals;
          }

          // ok, we can go faster!
          typename Solver::Type fc = (2 * rho - 1);
          typename Solver::Type alpha = std::min(_settings.maxScaleFactor, 1 - fc * fc * fc);
          alpha = std::max(_settings.minScaleFactor, alpha);
          mu = mu * alpha;
          curV = _settings.initV;

        }
        else {
          // no, we have to roll back
          op.rollbackSolution();
          mu = mu * curV;
          curV = curV * _settings.penaltyV;

        }

        subiter++;
        if (subiter >= _settings.maxNumSubIt) {
          return LevenbergMarquardtTerminationReason::MaxSubIterationReached;
        }

        toRepeat = (rho <= 0 || !solveOk);
      } while (toRepeat);

      // compute jacobians with current solution
      op.update(true);

      // and backup for next iteration
      op.backupSolution();
      oldChi2 = op.hessian().chi2();

      // next iteration
      iter++;
    }



    return LevenbergMarquardtTerminationReason::MaxIterationReached;
  }

}