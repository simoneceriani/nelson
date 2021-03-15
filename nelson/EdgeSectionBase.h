#pragma once
#include "Global.h"

#include <cassert>

namespace nelson {

  template<class Section>
  class EdgeSectionBase {

    Section* _section;

  protected:
    void setSection(Section* section);

    Section& section() {
      assert(_section != nullptr);
      return *_section;
    }

  public:
    EdgeSectionBase();
    virtual ~EdgeSectionBase();

    const Section& section() const {
      assert(_section != nullptr);
      return *_section;
    }
  };

}