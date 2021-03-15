#pragma once
#include "EdgeSectionBase.h"

namespace nelson {

  template<class Section>
  EdgeSectionBase<Section>::EdgeSectionBase() :
    _section(nullptr)
  {

  }

  template<class Section>
  EdgeSectionBase<Section>::~EdgeSectionBase() {

  }

  template<class Section>
  void EdgeSectionBase<Section>::setSection(Section* section) {
    this->_section = section;
  }

}