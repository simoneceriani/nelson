#include "GaussNewton.h"
#include "GaussNewton.hpp"
#include <sstream>

#include <sstream>
#include <iomanip>

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
  
  GaussNewtownStats::GaussNewtownStats() {
  }

  GaussNewtownStats::~GaussNewtownStats() {

  }

  void GaussNewtownStats::reserve(int n) {
    _stats.reserve(n);
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

  //------------------------------------------------------------------------------------------------------

  GaussNewtownTimingStats::IterationTime::IterationTime()
    :
    evalTime(0),
    applyIncrementTime(0),
    computeIncrementTime(0), 
    overallTime(0)
  {

  }

  std::string GaussNewtownTimingStats::IterationTime::toString(const std::string& linePrefix) const {
    std::ostringstream s;
    s << std::fixed << std::setprecision(6);
    s << linePrefix << "overall time = " << overallTime.count() << std::endl;
    s << linePrefix << "-- compute inc    = " << computeIncrementTime.count() << std::endl;
    s << linePrefix << "-- apply inc    = " << applyIncrementTime.count() << std::endl;
    s << linePrefix << "-- eval time    = " << evalTime.count() << std::endl;
    return s.str();
  }
  

  GaussNewtownTimingStats::GaussNewtownTimingStats() :
    _firstEvalTime(0),
    _solverInitTime(0)

  {

  }

  GaussNewtownTimingStats::~GaussNewtownTimingStats() {

  }

  void GaussNewtownTimingStats::reserve(int n) {
    _stats.reserve(n);
  }

  void GaussNewtownTimingStats::addIteration(const IterationTime& it)
  {
    _stats.push_back(it);
  }

  std::string GaussNewtownTimingStats::toString(const std::string& linePrefix) const {
    std::ostringstream s;
    s << std::fixed << std::setprecision(6) << std::endl;
    s << linePrefix << "first evaluation " << _firstEvalTime.count() << std::endl;
    s << linePrefix << "solver init " << _solverInitTime.count() << std::endl;
    for (int i = 0; i < _stats.size(); i++) {
      s << linePrefix << "-- ITER " << i << " -- " << std::endl;
      s << linePrefix << _stats[i].toString("  ");
    }
    return s.str();
  }




}