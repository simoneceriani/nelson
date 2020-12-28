#pragma once
#include "Global.h"

#include <cassert>

namespace nelson {

  template<class Section>
  class EdgeSingleSectionBase {

    Section* _section;

  protected:
    void setSection(Section* section);

    Section& section() {
      assert(_section != nullptr);
      return *_section;
    }

  public:
    EdgeSingleSectionBase();
    virtual ~EdgeSingleSectionBase();

    const Section& section() const {
      assert(_section != nullptr);
      return *_section;
    }
  };

}