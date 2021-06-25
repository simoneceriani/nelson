#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

#include "SingleSectionHessian.h"
#include "BaseSection.h"

#include "EdgeInterface.h"
#include "EdgeUnary.h"
#include "EdgeNary.h"
#include "EdgeBNary.h"
#include "EdgeBinary.h"

#include <memory>
#include <vector>
#include <array>
#include <map>

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

    constexpr int parameterSizePermuted(const Eigen::VectorXi& user2interal) const {
      return B;
    }
  };

  template<int B>
  class BaseNumSizeParameters<B, mat::Dynamic> {
  public:
    constexpr int parameterSize() const {
      return B;
    }
    virtual int numParameters() const = 0;

    int parameterSizePermuted(const Eigen::VectorXi& user2interal) const {
      return B;
    }

  };

  template<int NB>
  class BaseNumSizeParameters<mat::Dynamic, NB> {
  public:
    virtual int parameterSize() const = 0;
    constexpr int numParameters() const {
      return NB;
    }
    int parameterSizePermuted(const Eigen::VectorXi& user2interal) const {
      return parameterSize();
    }

  };

  template<>
  class BaseNumSizeParameters<mat::Dynamic, mat::Dynamic> {
  public:
    virtual int parameterSize() const = 0;
    virtual int numParameters() const = 0;
    int parameterSizePermuted(const Eigen::VectorXi& user2interal) const {
      return parameterSize();
    }

  };

  template<int NB>
  class BaseNumSizeParameters<mat::Variable, NB> {
  public:
    virtual const std::vector<int>& parameterSize() const = 0;
    constexpr int numParameters() const {
      assert(parameterSize().size() == NB);
      return NB;
    }

    std::vector<int> parameterSizePermuted(const Eigen::VectorXi& user2internal) const {
      const auto& parsize = this->parameterSize();
      std::vector<int> ret(user2internal.size());
      for (int i = 0; i < ret.size(); i++) {
        ret[user2internal(i)] = parsize[i];
      }
      return ret;
    }
  };

  template<>
  class BaseNumSizeParameters<mat::Variable, mat::Dynamic> {
  public:
    virtual const std::vector<int>& parameterSize() const = 0;
    int numParameters() const {
      return parameterSize().size();
    }
    std::vector<int> parameterSizePermuted(const Eigen::VectorXi& user2internal) const {
      const auto& parsize = this->parameterSize();
      std::vector<int> ret(user2internal.size());
      for (int i = 0; i < ret.size(); i++) {
        ret[user2internal(i)] = parsize[i];
      }
      return ret;
    }
  };


  //--------------------------------------------------------------------------------

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB = mat::Dynamic>
  class SingleSection : public BaseNumSizeParameters<B, NB>, public BaseSection {
  public:

    using Hessian = SingleSectionHessian<matTypeV, T, B, NB>;
    using HessianVecType = typename Hessian::VecType;
    using ParameterType = ParT;
  private:

    // once the number of params is known (parametersReady() called), _sparsityPattern is allocated and _edgeSetter too, ready to receive edges
    std::shared_ptr<mat::SparsityPattern<mat::ColMajor>> _sparsityPattern;

    std::vector<std::map<int, ListWithCount>> _edgeSetterComputer;

    Eigen::Matrix<int, NB, 1> _user2internalIndexes;

    Hessian _hessian;


    BaseSectionSettings _settings;

    void updateHessianBlocks();
    void evaluateEdges(bool hessian);

    void setChi2(double v) override {
      _hessian.setChi2(v);
    }

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

    const Eigen::Matrix<int, NB, 1>& user2internalIndexes() const {
      return _user2internalIndexes;
    } 

    void setUser2InternalIndexes(const Eigen::Matrix<int, NB, 1>& v);

    void permuteAMD();

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

    void update(bool hessian);

    BaseSectionSettings& settings() {
      return _settings;
    }
    const BaseSectionSettings& settings() const {
      return _settings;
    }

    T computeRhoChi2Change(T mu, const HessianVecType& incV, T oldChi2) const {
      T newChi2 = hessian().chi2();
      T num = (oldChi2 - newChi2);
      T den = incV.mat().transpose() * (mu * incV.mat() - hessian().b().mat());
      //
      return num / den;

    }

    //-------------------------------------------------------------------------------------------
    struct EdgeUnaryAdapter {

      using ParameterType = ParT;
      using HBlockType = typename Hessian::MatTraits::MatrixType::BlockType;
      using BSegmentType = typename Hessian::VecType::SegmentType;

      static const ParameterType& parameter(const Derived& section, NodeId id) {
        return section.parameter(id);
      }
      static HBlockType HBlock(Derived& section, int uid) {
        return section.hessianBlockByUID(uid);
      }
      static BSegmentType bSegment(Derived& section, int par_id) {
        return section.bVectorSegment(section.user2internalIndexes()(par_id));
      }

    };

    template<class EdgeDerived>
    using EdgeUnary = EdgeUnarySectionBaseCRPT<Derived, EdgeUnaryAdapter, EdgeDerived>;

    template<class EdgeDerived>
    void addEdge(NodeId i, EdgeUnary<EdgeDerived>* e);


    //-------------------------------------------------------------------------------------------
    struct EdgeBinaryAdapter {

      using Parameter_1_Type = ParameterType;
      using Parameter_2_Type = ParameterType;
      
      using H_11_BlockType = typename Hessian::MatTraits::MatrixType::BlockType;
      using B_1_SegmentType = typename Hessian::VecType::SegmentType;

      using H_12_BlockType = typename Hessian::MatTraits::MatrixType::BlockType;

      using H_22_BlockType = typename Hessian::MatTraits::MatrixType::BlockType;
      using B_2_SegmentType = typename Hessian::VecType::SegmentType;

      static const Parameter_1_Type& parameter1(const Derived& section, NodeId id) {
        return section.parameter(id);
      }
      static const Parameter_2_Type& parameter2(const Derived& section, NodeId id) {
        return section.parameter(id);
      }

      static H_11_BlockType H_11_Block(Derived& section, int uid) {
        return section.hessianBlockByUID(uid);
      }
      static B_1_SegmentType b_1_Segment(Derived& section, int par_id) {
        return section.bVectorSegment(section.user2internalIndexes()(par_id));
      }

      static H_12_BlockType H_12_Block(Derived& section, int uid) {
        return section.hessianBlockByUID(uid);
      }

      static H_22_BlockType H_22_Block(Derived& section, int uid) {
        return section.hessianBlockByUID(uid);
      }
      static B_2_SegmentType b_2_Segment(Derived& section, int par_id) {
        return section.bVectorSegment(section.user2internalIndexes()(par_id));
      }

    };

    template<class EdgeDerived>
    using EdgeBinary = EdgeBinarySectionBaseCRPT<Derived, EdgeBinaryAdapter, EdgeDerived>;

    template<class EdgeDerived>
    void addEdge(NodeId i, NodeId j, EdgeBinary<EdgeDerived>* e);

    template<class EdgeDerived, int N>
    using EdgeNary = EdgeNarySectionBaseCRPT<Derived, N, EdgeUnaryAdapter, EdgeDerived>;

    template<class EdgeDerived, int N>
    void addEdge(const std::array<NodeId, N>& ids, EdgeNary<EdgeDerived, N> *e);
    
    template<class EdgeDerived>
    void addEdge(const std::vector<NodeId>& ids, EdgeNary<EdgeDerived, mat::Dynamic> *e);


    template<class EdgeDerived, int N1, int N2>
    using EdgeBNary = EdgeBNarySectionBaseCRPT<Derived, N1, N2, EdgeBinaryAdapter, EdgeDerived>;

    // generic
    template<class EdgeDerived, int N1, int N2, class Container1, class Container2>
    void addEdgeT(const Container1& ids1, const Container2& ids2, EdgeBNary<EdgeDerived, N1, N2> *e);    

    template<class EdgeDerived, int N1, int N2>
    void addEdge(const std::array<NodeId, N1>& ids1, const std::array<NodeId, N2>& ids2, EdgeBNary<EdgeDerived, N1, N2> *e);    
    template<class EdgeDerived, int N1>
    void addEdge(const std::array<NodeId, N1>& ids1, const std::vector<NodeId>& ids2, EdgeBNary<EdgeDerived, N1, mat::Dynamic> *e);
    template<class EdgeDerived, int N2>
    void addEdge(const std::vector<NodeId>& ids1, const std::array<NodeId, N2>& ids2, EdgeBNary<EdgeDerived, mat::Dynamic, N2> *e);
    template<class EdgeDerived>
    void addEdge(const std::vector<NodeId>& ids1, const std::vector<NodeId>& ids2, EdgeBNary<EdgeDerived, mat::Dynamic, mat::Dynamic> *e);


  };

}
