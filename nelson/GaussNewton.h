#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
#include "SolverTraits.h"

#include <vector>
#include <string>

namespace nelson {

  struct GaussNewtonSettings {
    int maxNumIt;
    int minNumIt;
    double epsBVector;
    double epsIncSquare;
    double epsChi2;
    double absLambda;
    double relLambda;

    GaussNewtonSettings();
    virtual ~GaussNewtonSettings();

  };

  enum class GaussNewtonTerminationReason {
    SuccessAtStart,
    SuccessSmallIncrement,
    SuccessEpsB,
    SuccessSmallResiduals,
    SolverFailure,
    MaxIterationReached
  };

  class GaussNewtonUtils {
    static const std::vector<std::string> _terminationReasonStrings;
  public:

    static bool success(GaussNewtonTerminationReason lm);
    static std::string toString(GaussNewtonTerminationReason lm);

  };

  template<int solverTypeV, int matTypeV, class T, int B, int NB = mat::Dynamic>
  class GaussNewton {

    typename SolverTraits<solverTypeV>::template Solver<matTypeV, T, B, NB> _solver;

    GaussNewtonSettings _settings;
  public:
    GaussNewton();
    GaussNewton(const GaussNewtonSettings & settings);
    virtual ~GaussNewton();

    GaussNewtonSettings& settings() { return _settings; }
    const GaussNewtonSettings& settings() const { return _settings; }


    template<class OptimizationProblem>
    GaussNewtonTerminationReason solve(OptimizationProblem& op);
  };

}