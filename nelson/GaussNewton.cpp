#include "GaussNewton.h"
#include "GaussNewton.hpp"
#include <sstream>

namespace nelson {



  GaussNewtonSettings::GaussNewtonSettings() :
    maxNumIt(100),
    minNumIt(0),
    epsBVector(1e-15),
    epsIncSquare(1e-15 * 1e-15),
    epsChi2(1e-15),
    absLambda(0),
    relLambda(0)
  {

  }

  GaussNewtonSettings::~GaussNewtonSettings() {

  }

  //------------------------------------------------------------------------------------------------------

  const std::vector<std::string> GaussNewtonUtils::_terminationReasonStrings = {
  "SuccessAtStart",
  "SuccessSmallIncrement",
  "SuccessEpsB",
  "SuccessSmallResiduals",
  "SolverFailure",
  "MaxIterationReached"
  };

  bool GaussNewtonUtils::success(GaussNewtonTerminationReason lm) {
    return
      lm == GaussNewtonTerminationReason::SuccessAtStart ||
      lm == GaussNewtonTerminationReason::SuccessEpsB ||
      lm == GaussNewtonTerminationReason::SuccessSmallIncrement ||
      lm == GaussNewtonTerminationReason::SuccessSmallResiduals;
  }

  std::string GaussNewtonUtils::toString(GaussNewtonTerminationReason lm) {
    return _terminationReasonStrings[int(lm)];
  }

  //------------------------------------------------------------------------------------------------------
  
  GaussNewtownStats::GaussNewtownStats(int reserve) {
    _stats.reserve(reserve);
  }

  GaussNewtownStats::~GaussNewtownStats() {

  }

  void GaussNewtownStats::addIteration(int it, double chi2) {
    _stats.push_back(Stats{ it,chi2 });
  }

  std::string GaussNewtownStats::toString() const {
    std::ostringstream s;
    for (int i = 0; i < _stats.size(); i++) {
      s << "it = " << _stats[i].it << ", chi2 = " << _stats[i].chi2 << std::endl;
    }
    return s.str();
  }



}