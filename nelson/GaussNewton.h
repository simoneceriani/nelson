#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
#include "SolverCholeskySchur.h"

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
    GaussNewtownStats();
    virtual ~GaussNewtownStats();

    void reserve(int n);

    void addIteration(int it, double chi2);
    std::string toString() const;
  };

  //------------------------------------------------------------------------------------------------------

  template<class Solver>
  class GaussNewton {

    Solver _solver;

    GaussNewtonSettings _settings;
    int iter;

    GaussNewtownStats _stats;
  public:

    using Utils = GaussNewtonUtils;
    using Settings = GaussNewtonSettings;
    using Stats = GaussNewtownStats;

    GaussNewton();
    GaussNewton(const GaussNewtonSettings & settings);
    virtual ~GaussNewton();

    GaussNewtonSettings& settings() { return _settings; }
    const GaussNewtonSettings& settings() const { return _settings; }

    template<typename RetType = typename Solver::Settings &>
    inline std::enable_if_t<Solver::hasSettings, RetType> solverSettings() { return _solver.settings(); }

    template<typename RetType = const typename Solver::Settings&>
    inline std::enable_if_t<Solver::hasSettings, RetType> solverSettings() const { return _solver.settings(); }

    const GaussNewtownStats stats() const { return _stats; }

    template<class OptimizationProblem>
    GaussNewtonTerminationReason solve(OptimizationProblem& op);

    int numIterations() const {
      return iter;
    }
  };

}