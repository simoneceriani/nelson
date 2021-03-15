#pragma once
#include "Global.h"

#include "ParallelExecHelper.h"

#include <memory>
#include <vector>
#include <forward_list>

#include "EdgeInterface.h"

namespace nelson {

  struct BaseSectionSettings {
    ParallelExecSettings edgeEvalParallelSettings;
    ParallelExecSettings hessianUpdateParallelSettings;
  };

  class BaseSection {

  protected:

    struct SetterComputer {
      std::unique_ptr<EdgeUIDSetterInterface> setter;
      std::unique_ptr<EdgeHessianUpdater> computer;
      SetterComputer(EdgeUIDSetterInterface* s, EdgeHessianUpdater* c) : setter(s), computer(c) {
      }
      SetterComputer() {}
    };

    struct ListWithCount {
      std::forward_list<SetterComputer> list;
      int size;
      ListWithCount() : size(0) {

      }
    };

    std::vector<std::unique_ptr<EdgeInterface>> _edgesVector;
    
    // outer size is the number of independent computation, safe to be computed in parallel, inside they have to go sequential (or parallel but with reduction)
    std::vector<std::vector<std::unique_ptr<EdgeHessianUpdater>>> _computationUnits;

    void updateHessianBlocks(const ParallelExecSettings & hessianUpdateParallelSettings);
    void evaluateEdges(const ParallelExecSettings& edgeEvalParallelSettings, bool hessian);

    virtual void setChi2(double v) = 0;

  public:

    void reserveEdges(int n) {
      _edgesVector.reserve(n);
    }

    int numEdges() const {
      return int(_edgesVector.size());
    }

  };

}