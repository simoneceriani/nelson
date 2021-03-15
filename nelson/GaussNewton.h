#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
#include "SolverCholeskySchur.h"
#include "SolverDiagonalBlocksInverseSchur.h"
#include "SolverDiagonalBlocksInverseWWtMultSchur.h"

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

  class GaussNewtownTimingStats {
  public:
    struct IterationTime {
      std::chrono::duration<double> evalTime;
      std::chrono::duration<double> applyIncrementTime;
      std::chrono::duration<double> computeIncrementTime;
      std::chrono::duration<double> overallTime;
      IterationTime();
      std::string toString(const std::string& linePrefix = "") const;
    };

  private:

    std::vector<IterationTime> _stats;
    std::chrono::duration<double> _firstEvalTime;
    std::chrono::duration<double> _solverInitTime;

  public:
    GaussNewtownTimingStats();
    virtual ~GaussNewtownTimingStats();

    void reserve(int n);

    void setFirstEvalTime(const std::chrono::duration<double>& d) { _firstEvalTime = d; };
    void setSolverInitTime(const std::chrono::duration<double>& d) { _solverInitTime = d; }

    void addIteration(const IterationTime& it);

    std::string toString(const std::string& linePrefix = "") const;
  };


  //------------------------------------------------------------------------------------------------------

  template<class Solver>
  class GaussNewton {

    Solver _solver;

    GaussNewtonSettings _settings;
    int iter;

    GaussNewtownStats _stats;
    GaussNewtownTimingStats _timingStats;
  public:

    using Utils = GaussNewtonUtils;
    using Settings = GaussNewtonSettings;
    using Stats = GaussNewtownStats;
    using TimingStats = GaussNewtownTimingStats;

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
    const GaussNewtownTimingStats timingStats() const { return _timingStats; }

    const Solver& solver() const { return _solver; }

    template<class OptimizationProblem>
    GaussNewtonTerminationReason solve(OptimizationProblem& op);

    int numIterations() const {
      return iter;
    }
  };

}