#pragma once
#include "EdgeBinary.h"

namespace nelson {

  template<class Section>
  EdgeBinarySectionBase<Section>::EdgeBinarySectionBase() :
    EdgeBinaryBase(),
    EdgeSectionBase<Section>()
  {

  }

  template<class Section>
  EdgeBinarySectionBase<Section>::~EdgeBinarySectionBase() {

  }


}