#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
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

  template<int solverTypeV, class HessianTraits>
  class LevenbergMarquardt {

    typename SolverTraits<solverTypeV>::template Solver<HessianTraits> _solver;

    LevenbergMarquardtSettings _settings;
    int iter;
    int subiter;

    LevenbergMarquardtStats _stats;
  public:
    LevenbergMarquardt();
    LevenbergMarquardt(const LevenbergMarquardtSettings& settings);
    virtual ~LevenbergMarquardt();

    LevenbergMarquardtSettings& settings() { return _settings; }
    const LevenbergMarquardtSettings& settings() const { return _settings; }

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