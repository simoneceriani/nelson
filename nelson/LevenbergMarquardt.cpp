#include "LevenbergMarquardt.h"
#include "LevenbergMarquardt.hpp"

namespace nelson {

  LevenbergMarquardtSettings::LevenbergMarquardtSettings() 
    : 
    maxNumIt(100),
    maxNumSubIt(10),
    minNumIt(0),
    epsBVector(1e-15),
    epsIncSquare(1e-15 * 1e-15),
    epsChi2(1e-15),
    tau(1e-3),
    penaltyV(2.0),
    initV(2.0),
    maxScaleFactor(2.0 / 3.0),
    minScaleFactor(1.0 / 3.0)
  {

  }
  LevenbergMarquardtSettings::~LevenbergMarquardtSettings() {}

  //------------------------------------------------------------------------------------------------------

  const std::vector<std::string> LevenbergMarquardtUtils::_terminationReasonStrings = {
  "SuccessAtStart",
  "SuccessSmallIncrement",
  "SuccessEpsB",
  "SuccessSmallResiduals",
  "SolverFailure",
  "MaxIterationReached",
  "MaxSubIterationReached"
  };

  bool LevenbergMarquardtUtils::success(LevenbergMarquardtTerminationReason lm) {
    return
      lm == LevenbergMarquardtTerminationReason::SuccessAtStart ||
      lm == LevenbergMarquardtTerminationReason::SuccessEpsB ||
      lm == LevenbergMarquardtTerminationReason::SuccessSmallIncrement ||
      lm == LevenbergMarquardtTerminationReason::SuccessSmallResiduals;
  }

  std::string LevenbergMarquardtUtils::toString(LevenbergMarquardtTerminationReason lm) {
    return _terminationReasonStrings[int(lm)];
  }

  //------------------------------------------------------------------------------------------------------

  LevenbergMarquardtStats::LevenbergMarquardtStats() {

  }
  LevenbergMarquardtStats::~LevenbergMarquardtStats() {

  }

  void LevenbergMarquardtStats::reserve(int reserveOut, int reserveInStats) {
    _reserveInStats = reserveOut;
    _stats.reserve(reserveOut);
  }

  void LevenbergMarquardtStats::addIteration(int it) {
    _stats.push_back(OutStats(it, _reserveInStats));
  }

  void LevenbergMarquardtStats::addSubIteration(int subit, double chi2) {
    assert(_stats.size() > 0);
    _stats.back().subit.push_back(InStats{ subit,chi2 });
  }
  std::string LevenbergMarquardtStats::toString() const {
    std::ostringstream s;
    for (int i = 0; i < _stats.size(); i++) {
      s << "it = " << _stats[i].it << std::endl;
      for (int j = 0; j < _stats[i].subit.size(); j++) {
        s << "-- subit = " << _stats[i].subit[j].subit << ", chi2 = " << _stats[i].subit[j].chi2 << std::endl;
      }
    }
    return s.str();

  }

}
