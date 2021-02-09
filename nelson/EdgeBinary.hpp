#pragma once
#include "EdgeBinary.h"

namespace nelson {

  template<class Section>
  EdgeBinarySingleSection<Section>::EdgeBinarySingleSection() :
    EdgeBinaryBase(),
    EdgeSectionBase<Section>()
  {

  }

  template<class Section>
  EdgeBinarySingleSection<Section>::~EdgeBinarySingleSection() {

  }


}