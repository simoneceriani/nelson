#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

#include "SingleSectionHessian.h"
#include "ParallelExecHelper.h"

#include "EdgeInterface.h"
#include "EdgeUnary.h"
#include "EdgeBinary.h"

#include <memory>
#include <vector>
#include <map>
#include <forward_list>

namespace nelson {

  template<int B, int NB>
  class BaseNumSizeParameters {
  public:
    constexpr int parameterSize() const {
      return B;
    }
    constexpr int numParameters() const {
      return NB;
    }
  };

  template<int B>
  class BaseNumSizeParameters<B, mat::Dynamic> {
  public:
    constexpr int parameterSize() const {
      return B;
    }
    virtual int numParameters() const = 0;
  };

  template<int NB>
  class BaseNumSizeParameters<mat::Dynamic, NB> {
  public:
    virtual int parameterSize() const = 0;
    constexpr int numParameters() const {
      return NB;
    }
  };

  template<>
  class BaseNumSizeParameters<mat::Dynamic, mat::Dynamic> {
  public:
    virtual int parameterSize() const = 0;
    virtual int numParameters() const = 0;
  };

  template<int NB>
  class BaseNumSizeParameters<mat::Variable, NB> {
  public:
    virtual const std::vector<int>& parameterSize() const = 0;
    constexpr int numParameters() const {
      assert(parameterSize().size() == NB);
      return NB;
    }
  };

  template<>
  class BaseNumSizeParameters<mat::Variable, mat::Dynamic> {
  public:
    virtual const std::vector<int>& parameterSize() const = 0;
    int numParameters() const {
      return parameterSize().size();
    }
  };


  //--------------------------------------------------------------------------------

  struct SingleSectionSettings {
    ParallelExecSettings edgeEvalParallelSettings;
    ParallelExecSettings hessianUpdateParallelSettings;
  };

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SingleSection : public BaseNumSizeParameters<B, NB> {
  public:

    using Hessian = SingleSectionHessian<matTypeV, T, B, NB>;
    using HessianVecType = typename Hessian::VecType;
    using ParameterType = ParT;
  private:

    // once the number of params is known (parametersReady() called), _sparsityPattern is allocated and _edgeSetter too, ready to receive edges
    std::shared_ptr<mat::SparsityPattern<mat::ColMajor>> _sparsityPattern;

    struct SetterComputer {
      std::unique_ptr<EdgeUIDSetterInterface> setter;
      std::unique_ptr<EdgeHessianUpdater> computer;
      SetterComputer(EdgeUIDSetterInterface* s, EdgeHessianUpdater* c) : setter(s), computer(c) {     
      }
      SetterComputer(){}
    };

    struct ListWithCount {
      std::forward_list<SetterComputer> list;
      int size;
      ListWithCount() : size(0) {

      }
    };

    std::vector<std::map<int, ListWithCount>> _edgeSetterComputer;

    Hessian _hessian;


    std::forward_list<std::unique_ptr<EdgeInterface>> _edges;
    int _edgesCount;

    std::vector<std::unique_ptr<EdgeInterface>> _edgesVector;

    // outer size is the number of independent computation, safe to be computed in parallel, inside they have to go sequential (or parallel but with reduction)
    std::vector<std::vector<std::unique_ptr<EdgeHessianUpdater>>> _computationUnits;

    SingleSectionSettings _settings;

    void updateHessianBlocks();
    void evaluateEdges(bool hessian);

  public:

    constexpr int matType() const { return matTypeV; }

    SingleSection();
    virtual ~SingleSection();

    virtual const ParT& parameter(NodeId i) const = 0;
    virtual ParT& parameter(NodeId i) = 0;

    virtual int numFixedParameters() const {
      // override if have fixed parameters
      return 0;
    }

    // virtual int numParameters() const = 0; // not defined here, but in base class BaseNumParameters<NB>, only if not fixed size (NB)!
    // virtual const std::vector<int>& | int parameterSize() const = 0; // not defined here, but in base class BaseParameterSize<B>, only if not fixed size !


    void parametersReady(); // client has to call this method when numParameters() is known
    void structureReady(); // client has to call this method when all edges have been added

    const mat::SparsityPattern<mat::ColMajor >& sparsityPattern() const {
      assert(_sparsityPattern != nullptr);
      return *_sparsityPattern;
    }

    const Hessian& hessian() const {
      return _hessian;
    }

    // friend only for gauss newton??
    Hessian& hessian() {
      return _hessian;
    }

    typename Hessian::MatTraits::MatrixType::BlockType hessianBlockByUID(int uid) {
      return _hessian.H().blockByUID(uid);
    }

    typename Hessian::VecType::SegmentType bVectorSegment(int pid) {
      return _hessian.b().segment(pid);
    }

    void addEdge(NodeId i, EdgeUnarySingleSection<Derived>* e);
    void addEdge(NodeId i, NodeId j, EdgeBinarySingleSection<Derived>* e);
    //void addEdge(int i, int j, int k/*, EdgeTernary* e*/);
    //template<int N>
    //void addEdge(const std::array<int, N>& ids/*, EdgeNAry * e*/);

    void update(bool hessian);

    SingleSectionSettings& settings() {
      return _settings;
    }
    const SingleSectionSettings& settings() const {
      return _settings;
    }
  };

}