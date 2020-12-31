#include "GaussNewton.h"
#include "GaussNewton.hpp"

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
  "CholeskyFailure",
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


}