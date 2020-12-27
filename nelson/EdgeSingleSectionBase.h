#pragma once
#include "Global.h"

#include <cassert>

namespace nelson {

  template<class Section>
  class EdgeSingleSectionBase {

    const Section* _section;

  protected:
    void setSection(const Section* section);

  public:
    EdgeSingleSectionBase();
    virtual ~EdgeSingleSectionBase();

    const Section& section() const {
      assert(_section != nullptr);
      return *_section;
    }
  };

}