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

  //------------------------------------------------------------------------------------------------------

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

  //------------------------------------------------------------------------------------------------------

  class GaussNewtownStats {
  public:
    struct Stats {
      int it;
      double chi2;
    };

  private:
    std::vector<Stats> _stats;
    
  public:
    GaussNewtownStats(int reserve);
    virtual ~GaussNewtownStats();

    void addIteration(int it, double chi2);
    std::string toString() const;
  };

  //------------------------------------------------------------------------------------------------------

  template<int solverTypeV, int matTypeV, class T, int B, int NB = mat::Dynamic>
  class GaussNewton {

    typename SolverTraits<solverTypeV>::template Solver<matTypeV, T, B, NB> _solver;

    GaussNewtonSettings _settings;
    int iter;

    GaussNewtownStats _stats;
  public:
    GaussNewton();
    GaussNewton(const GaussNewtonSettings & settings);
    virtual ~GaussNewton();

    GaussNewtonSettings& settings() { return _settings; }
    const GaussNewtonSettings& settings() const { return _settings; }

    const GaussNewtownStats stats() const { return _stats; }

    template<class OptimizationProblem>
    GaussNewtonTerminationReason solve(OptimizationProblem& op);

    int numIterations() const {
      return iter;
    }
  };

  template<int solverTypeV, class HessianTraits>
  using GaussNewtonHessianTraits = GaussNewton< solverTypeV, HessianTraits::matType, typename HessianTraits::Type, HessianTraits::B, HessianTraits::NB>;

}