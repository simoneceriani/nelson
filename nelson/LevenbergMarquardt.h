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

  struct LevenbergMarquardtSettings {
    int maxNumIt;
    int maxNumSubIt;
    int minNumIt;
    double epsBVector;
    double epsIncSquare;
    double epsChi2;
    double tau;
    double penaltyV;
    double initV;
    double maxScaleFactor;
    double minScaleFactor;

    LevenbergMarquardtSettings();
    virtual ~LevenbergMarquardtSettings();

  };

  enum class LevenbergMarquardtTerminationReason {
    SuccessAtStart,
    SuccessSmallIncrement,
    SuccessEpsB,
    SuccessSmallResiduals,
    SolverFailure,
    MaxIterationReached,
    MaxSubIterationReached
  };

  class LevenbergMarquardtUtils {
    static const std::vector<std::string> _terminationReasonStrings;
  public:

    static bool success(LevenbergMarquardtTerminationReason lm);
    static std::string toString(LevenbergMarquardtTerminationReason lm);

  };

  //------------------------------------------------------------------------------------------------------

  class LevenbergMarquardtStats {
  public:
    struct InStats {
      int subit;
      double chi2;
    };
    struct OutStats {
      int it;
      std::vector<InStats> subit;
      OutStats(int it, int reserveIn) : it(it) {
        subit.reserve(reserveIn);
      }
    };

  private:
    std::vector<OutStats> _stats;
    int _reserveInStats;
  
  public:
    LevenbergMarquardtStats();
    virtual ~LevenbergMarquardtStats();

    void reserve(int reserveOut, int reserveInStats);

    void addIteration(int it);
    void addSubIteration(int subit, double chi2);
    std::string toString() const;

  };

  //------------------------------------------------------------------------------------------------------

  template<class Solver>
  class LevenbergMarquardt {

    Solver _solver;

    LevenbergMarquardtSettings _settings;
    int iter;
    int subiter;

    LevenbergMarquardtStats _stats;
  public:

    using Utils = LevenbergMarquardtUtils;
    using Settings = LevenbergMarquardtSettings;
    using Stats = LevenbergMarquardtStats;

    LevenbergMarquardt();
    LevenbergMarquardt(const LevenbergMarquardtSettings& settings);
    virtual ~LevenbergMarquardt();

    LevenbergMarquardtSettings& settings() { return _settings; }
    const LevenbergMarquardtSettings& settings() const { return _settings; }

    template<typename RetType = typename Solver::Settings&>
    inline std::enable_if_t<Solver::hasSettings, RetType> solverSettings() { return _solver.settings(); }

    template<typename RetType = const typename Solver::Settings&>
    inline std::enable_if_t<Solver::hasSettings, RetType> solverSettings() const { return _solver.settings(); }

    const Solver& solver() const { return _solver; }

    const LevenbergMarquardtStats stats() const { return _stats; }

    template<class OptimizationProblem>
    LevenbergMarquardtTerminationReason solve(OptimizationProblem& op);

    int numIterations() const {
      return iter;
    }
    int numSubIterations() const {
      return subiter;
    }
  };


}