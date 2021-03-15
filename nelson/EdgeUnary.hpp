#pragma once
#include "EdgeUnary.h"
#include "EdgeSectionBase.hpp"

namespace nelson {

  template<class Section>
  EdgeUnarySectionBase<Section>::EdgeUnarySectionBase() :
    EdgeUnaryBase(),
    EdgeSectionBase<Section>()
  {

  }

  template<class Section>
  EdgeUnarySectionBase<Section>::~EdgeUnarySectionBase() {

  }


}