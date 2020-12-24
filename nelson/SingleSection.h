#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

#include "SingleSectionHessian.h"

#include "EdgeInterface.h"
#include <memory>
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
    virtual const std::vector<int> & parameterSize() const = 0;
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

  template<class ParT, int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SingleSection : public BaseNumSizeParameters<B, NB> {
  public:

    using Hessian = SingleSectionHessian<matTypeV, T, B, NB>;
  private:

    Hessian _hessian;
    std::forward_list<std::unique_ptr<EdgeInterface<T>>> _edges;

    // outer size is the number of independent computation, safe to be computed in parallel, inside they have to go sequential or parallel but with reduction
    //std::vector<std::forward_list<EdgeComputator*>> _computationUnits;
    std::unique_ptr<mat::SparsityPattern<mat::ColMajor>> _sparsityPattern;
  public:

    constexpr int matType() const { return matTypeV; }

    SingleSection();
    virtual ~SingleSection();
    
    virtual const ParT& parameter(int i) const = 0;
    virtual ParT& parameter(int i) = 0;

    // virtual int numParameters() = 0; // not defined here, but in base class BaseNumParameters<NB>, only if not fixed size (NB)!
    // virtual const std::vector<int>& | int parameterSize() = 0; // not defined here, but in base class BaseParameterSize<B>, only if not fixed size !

    void parametersReady(); // client has to call this method when numParameters() is known
    void structureReady(); // client has to call this method when all edges have been added

    const mat::SparsityPattern<mat::ColMajor >& sparsityPattern() const {
      assert(_sparsityPattern != nullptr);
      return *_sparsityPattern;
    }

    const Hessian& hessian() const {
      return _hessian;
    }

    void addEdge(int i/*, EdgeUnary* e*/);
    void addEdge(int i, int j/*, EdgeBinary* e*/);
    void addEdge(int i, int j, int k/*, EdgeTernary* e*/);
    template<int N>
    void addEdge(const std::array<int, N> & ids/*, EdgeNAry * e*/);
  };

}