#pragma once
#include "EdgeSingleSectionBase.h"

namespace nelson {

  template<class Section>
  EdgeSingleSectionBase<Section>::EdgeSingleSectionBase() :
    _section(nullptr)
  {

  }

  template<class Section>
  EdgeSingleSectionBase<Section>::~EdgeSingleSectionBase() {

  }

  template<class Section>
  void EdgeSingleSectionBase<Section>::setSection(Section* section) {
    this->_section = section;
  }

}