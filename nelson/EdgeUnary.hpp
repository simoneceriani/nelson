#pragma once
#include "EdgeUnary.h"
#include "EdgeSingleSectionBase.hpp"

namespace nelson {

  template<class Section>
  EdgeUnarySingleSection<Section>::EdgeUnarySingleSection() : 
    EdgeUnaryBase(),
    EdgeSingleSectionBase<Section>()
  {

  }

  template<class Section>
  EdgeUnarySingleSection<Section>::~EdgeUnarySingleSection() {

  }


}