#pragma once
#include "Global.h"

#include "mat/Global.h"
#include "mat/MatrixTypeTraits.h"
#include "mat/SparsityPattern.h"
#include "mat/VectorBlock.h"

#include "DoubleSectionHessian.h"
#include "BaseSection.h"

#include "EdgeInterface.h"
#include "EdgeUnary.h"
#include "EdgeBinary.h"

#include <memory>
#include <vector>
#include <map>

namespace nelson {

  template<int B, int NB>
  class BaseNumSizeParametersU {
  public:
    constexpr int parameterUSize() const {
      return B;
    }
    constexpr int numParametersU() const {
      return NB;
    }
    constexpr int parameterUSizePermuted(const Eigen::VectorXi& user2interal) const {
      return B;
    }

  };

  template<int B>
  class BaseNumSizeParametersU<B, mat::Dynamic> {
  public:
    constexpr int parameterUSize() const {
      return B;
    }
    virtual int numParametersU() const = 0;
    constexpr int parameterUSizePermuted(const Eigen::VectorXi& user2interal) const {
      return B;
    }

  };

  template<int NB>
  class BaseNumSizeParametersU<mat::Dynamic, NB> {
  public:
    virtual int parameterUSize() const = 0;
    constexpr int numParametersU() const {
      return NB;
    }
    int parameterUSizePermuted(const Eigen::VectorXi& user2interal) const {
      return parameterUSize();
    }

  };

  template<>
  class BaseNumSizeParametersU<mat::Dynamic, mat::Dynamic> {
  public:
    virtual int parameterUSize() const = 0;
    virtual int numParametersU() const = 0;
    int parameterUSizePermuted(const Eigen::VectorXi& user2interal) const {
      return parameterUSize();
    }
  };

  template<int NB>
  class BaseNumSizeParametersU<mat::Variable, NB> {
  public:
    virtual const std::vector<int>& parameterUSize() const = 0;
    constexpr int numParametersU() const {
      assert(this->parameterUSize().size() == NB);
      return NB;
    }
    std::vector<int> parameterUSizePermuted(const Eigen::VectorXi& user2internal) const {
      const auto& parsize = this->parameterUSize();
      std::vector<int> ret(user2internal.size());
      for (int i = 0; i < ret.size(); i++) {
        ret[user2internal(i)] = parsize[i];
      }
      return ret;
    }

  };

  template<>
  class BaseNumSizeParametersU<mat::Variable, mat::Dynamic> {
  public:
    virtual const std::vector<int>& parameterUSize() const = 0;
    int numParametersU() const {
      return parameterUSize().size();
    }
    std::vector<int> parameterUSizePermuted(const Eigen::VectorXi& user2internal) const {
      const auto& parsize = this->parameterUSize();
      std::vector<int> ret(user2internal.size());
      for (int i = 0; i < ret.size(); i++) {
        ret[user2internal(i)] = parsize[i];
      }
      return ret;
    }
  };

  //-----------------------------------------------------------------------------------------------------

  template<int B, int NB>
  class BaseNumSizeParametersV {
  public:
    constexpr int parameterVSize() const {
      return B;
    }
    constexpr int numParametersV() const {
      return NB;
    }
  };

  template<int B>
  class BaseNumSizeParametersV<B, mat::Dynamic> {
  public:
    constexpr int parameterVSize() const {
      return B;
    }
    virtual int numParametersV() const = 0;
  };

  template<int NB>
  class BaseNumSizeParametersV<mat::Dynamic, NB> {
  public:
    virtual int parameterVSize() const = 0;
    constexpr int numParametersV() const {
      return NB;
    }
  };

  template<>
  class BaseNumSizeParametersV<mat::Dynamic, mat::Dynamic> {
  public:
    virtual int parameterVSize() const = 0;
    virtual int numParametersV() const = 0;
  };

  template<int NB>
  class BaseNumSizeParametersV<mat::Variable, NB> {
  public:
    virtual const std::vector<int>& parameterVSize() const = 0;
    constexpr int numParametersV() const {
      assert(this->parameterVSize().size() == NB);
      return NB;
    }
  };

  template<>
  class BaseNumSizeParametersV<mat::Variable, mat::Dynamic> {
  public:
    virtual const std::vector<int>& parameterVSize() const = 0;
    int numParametersV() const {
      return parameterVSize().size();
    }
  };

  //-----------------------------------------------------------------------------------------------------

  template<class DoubleSection, class EdgeBinaryParameter1_T, class EdgeBinaryParameter2_T>
  struct EdgeSameSection {
    static constexpr bool value = true;
  };

  template<class DoubleSection>
  struct EdgeSameSection<DoubleSection, typename DoubleSection::EdgeBinaryParameter1_U, typename DoubleSection::EdgeBinaryParameter2_V> {
    static constexpr bool value = false;
  };

  //-----------------------------------------------------------------------------------------------------
  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv = mat::Dynamic, int NBVv = mat::Dynamic>
  class DoubleSection : public BaseNumSizeParametersU<BUv, NBUv>, public BaseNumSizeParametersV<BVv, NBVv>, public BaseSection {

  public:

    using Hessian = DoubleSectionHessian<matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>;
    using HessianVecTypeU = typename Hessian::VecTypeU;
    using HessianVecTypeV = typename Hessian::VecTypeV;

    using HessianMatricesT = typename Hessian::HessianMatricesT;
    using HessianVectorsT = typename Hessian::HessianVectorsT;;


    using ParameterTypeU = ParUT;
    using ParameterTypeV = ParVT;
  private:

    // once the number of params is known (parametersReady() called), _sparsityPattern is allocated and _edgeSetter too, ready to receive edges
    std::shared_ptr<mat::SparsityPattern<mat::ColMajor>> _sparsityPatternU, _sparsityPatternV;
    std::shared_ptr<mat::SparsityPattern<mat::RowMajor>> _sparsityPatternW;

    std::vector<std::map<int, ListWithCount>> _edgeSetterComputerU, _edgeSetterComputerV, _edgeSetterComputerW;

    Eigen::Matrix<int, NBUv, 1> _user2internalIndexesU;

    Hessian _hessian;  

    BaseSectionSettings _settings;

    void updateHessianBlocks();
    void evaluateEdges(bool hessian);

    void setChi2(double v) override {
      _hessian.setChi2(v);
    }

  public:

    constexpr int matTypeU() const { return matTypeUv; }
    constexpr int matTypeV() const { return matTypeVv; }
    constexpr int matTypeW() const { return matTypeWv; }

    DoubleSection();
    virtual ~DoubleSection();

    virtual const ParUT& parameterU(NodeId i) const = 0;
    virtual ParUT& parameterU(NodeId i) = 0;
    virtual const ParVT& parameterV(NodeId i) const = 0;
    virtual ParVT& parameterV(NodeId i) = 0;

    virtual int numFixedParametersU() const {
      // override if have fixed parameters
      return 0;
    }
    virtual int numFixedParametersV() const {
      // override if have fixed parameters
      return 0;
    }

    const Eigen::Matrix<int, NBUv, 1>& user2internalIndexesU() const {
      return _user2internalIndexesU;
    }

    void setUser2InternalIndexesU(const Eigen::Matrix<int, NBUv, 1>& v);

    void permuteAMD_SchurU();


    void parametersReady(); // client has to call this method when numParameters() is known
    void structureReady(); // client has to call this method when all edges have been added

    const mat::SparsityPattern<mat::ColMajor >& sparsityPatternU() const {
      assert(_sparsityPatternU != nullptr);
      return *_sparsityPatternU;
    }
    const mat::SparsityPattern<mat::ColMajor >& sparsityPatternV() const {
      assert(_sparsityPatternV != nullptr);
      return *_sparsityPatternV;
    }
    const mat::SparsityPattern<mat::RowMajor >& sparsityPatternW() const {
      assert(_sparsityPatternW != nullptr);
      return *_sparsityPatternW;
    }

    const Hessian& hessian() const {
      return _hessian;
    }

    // friend only for gauss newton??
    Hessian& hessian() {
      return _hessian;
    }

    typename Hessian::MatTraitsU::MatrixType::BlockType hessianUBlockByUID(int uid) {
      return _hessian.U().blockByUID(uid);
    }
    typename Hessian::MatTraitsV::MatrixType::BlockType hessianVBlockByUID(int uid) {
      return _hessian.V().blockByUID(uid);
    }
    typename Hessian::MatTraitsW::MatrixType::BlockType hessianWBlockByUID(int uid) {
      return _hessian.W().blockByUID(uid);
    }

    typename Hessian::VecTypeU::SegmentType bUVectorSegment(int pid) {
      return _hessian.bU().segment(pid);
    }
    typename Hessian::VecTypeV::SegmentType bVVectorSegment(int pid) {
      return _hessian.bV().segment(pid);
    }
   
    void update(bool hessian);

    BaseSectionSettings& settings() {
      return _settings;
    }
    const BaseSectionSettings& settings() const {
      return _settings;
    }

    Tv computeRhoChi2Change(Tv mu, const HessianVectorsT& incV, Tv oldChi2) const {
      Tv newChi2 = hessian().chi2();
      Tv num = (oldChi2 - newChi2);
      auto den = 
        incV.bU().mat().transpose() * (mu * incV.bU().mat() - hessian().b().bU().mat()) + 
        incV.bV().mat().transpose() * (mu * incV.bV().mat() - hessian().b().bV().mat())
        ;
      //
      assert(den.size() == 1);
      return num / den(0);

    }


    struct EdgeUnaryUAdapter {

      using ParameterType = ParameterTypeU;
      using HBlockType = typename Hessian::MatTraitsU::MatrixType::BlockType;
      using BSegmentType = typename Hessian::VecTypeU::SegmentType;

      static const ParameterType& parameter(const Derived& section, NodeId id) {
        return section.parameterU(id);
      }
      static HBlockType HBlock(Derived& section, int uid) {
        return section.hessianUBlockByUID(uid);
      }
      static BSegmentType bSegment(Derived& section, int par_id) {
        return section.bUVectorSegment(section.user2internalIndexesU()(par_id));
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer(Derived& section) {
        return section._edgeSetterComputerU;
      }

    };

    struct EdgeUnaryVAdapter {

      using ParameterType = ParameterTypeV;
      using HBlockType = typename Hessian::MatTraitsV::MatrixType::BlockType;
      using BSegmentType = typename Hessian::VecTypeV::SegmentType;

      static const ParameterType& parameter(const Derived& section, NodeId id) {
        return section.parameterV(id);
      }
      static HBlockType HBlock(Derived& section, int uid) {
        return section.hessianVBlockByUID(uid);
      }
      static BSegmentType bSegment(Derived& section, int par_id) {
        return section.bVVectorSegment(par_id);
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer(Derived& section) {
        return section._edgeSetterComputerV;
      }

    };
    template<class EdgeUnaryAdapter, class EdgeDerived>
    using EdgeUnary = EdgeUnarySectionBaseCRPT<Derived, EdgeUnaryAdapter, EdgeDerived>;

    template<class EdgeDerived>
    using EdgeUnaryU = EdgeUnary<EdgeUnaryUAdapter, EdgeDerived>;
    template<class EdgeDerived>
    using EdgeUnaryV = EdgeUnary<EdgeUnaryVAdapter, EdgeDerived>;

    template<class EdgeUnaryAdapter, class EdgeDerived>
    void addEdge(NodeId i, EdgeUnary<EdgeUnaryAdapter, EdgeDerived>* e);

    //-------------------------------------------------------------------------------------------
    struct EdgeBinaryParameter1_U {

      using Parameter_1_Type = ParameterTypeU;
      using H_11_BlockType = typename Hessian::MatTraitsU::MatrixType::BlockType;
      using B_1_SegmentType = typename Hessian::VecTypeU::SegmentType;

      static const Parameter_1_Type& parameter1(const Derived& section, NodeId id) {
        return section.parameterU(id);
      }
      static H_11_BlockType H_11_Block(Derived& section, int uid) {
        return section.hessianUBlockByUID(uid);
      }
      static B_1_SegmentType b_1_Segment(Derived& section, int par_id) {
        return section.bUVectorSegment(section.user2internalIndexesU()(par_id));
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer1(Derived& section) {
        return section._edgeSetterComputerU;
      }

    };
    struct EdgeBinaryParameter1_V {

      using Parameter_1_Type = ParameterTypeV;
      using H_11_BlockType = typename Hessian::MatTraitsV::MatrixType::BlockType;
      using B_1_SegmentType = typename Hessian::VecTypeV::SegmentType;

      static const Parameter_1_Type& parameter1(const Derived& section, NodeId id) {
        return section.parameterV(id);
      }
      static H_11_BlockType H_11_Block(Derived& section, int uid) {
        return section.hessianVBlockByUID(uid);
      }
      static B_1_SegmentType b_1_Segment(Derived& section, int par_id) {
        return section.bVVectorSegment(par_id);
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer1(Derived& section) {
        return section._edgeSetterComputerV;
      }

    };

    struct EdgeBinaryParameter2_U {

      using Parameter_2_Type = ParameterTypeU;
      using H_22_BlockType = typename Hessian::MatTraitsU::MatrixType::BlockType;
      using B_2_SegmentType = typename Hessian::VecTypeU::SegmentType;

      static const Parameter_2_Type& parameter2(const Derived& section, NodeId id) {
        return section.parameterU(id);
      }
      static H_22_BlockType H_22_Block(Derived& section, int uid) {
        return section.hessianUBlockByUID(uid);
      }
      static B_2_SegmentType b_2_Segment(Derived& section, int par_id) {
        return section.bUVectorSegment(section.user2internalIndexesU()(par_id));
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer2(Derived& section) {
        return section._edgeSetterComputerU;
      }

    };
    struct EdgeBinaryParameter2_V {

      using Parameter_2_Type = ParameterTypeV;
      using H_22_BlockType = typename Hessian::MatTraitsV::MatrixType::BlockType;
      using B_2_SegmentType = typename Hessian::VecTypeV::SegmentType;

      static const Parameter_2_Type& parameter2(const Derived& section, NodeId id) {
        return section.parameterV(id);
      }
      static H_22_BlockType H_22_Block(Derived& section, int uid) {
        return section.hessianVBlockByUID(uid);
      }
      static B_2_SegmentType b_2_Segment(Derived& section, int par_id) {
        return section.bVVectorSegment(par_id);
      }

      static std::vector<std::map<int, ListWithCount>>& edgeSetterComputer2(Derived& section) {
        return section._edgeSetterComputerV;
      }

    };
    struct EdgeBinaryHessian12_UU {

      using H_12_BlockType = typename Hessian::MatTraitsU::MatrixType::BlockType;

      static H_12_BlockType H_12_Block(Derived& section, int uid) {
        return section.hessianUBlockByUID(uid);
      }

      static ListWithCount & edgeSetterComputer12(Derived& section, int i, int j) {
        return section._edgeSetterComputerU[j][i];
      }

    };
    struct EdgeBinaryHessian12_UV {

      using H_12_BlockType = typename Hessian::MatTraitsW::MatrixType::BlockType;

      static H_12_BlockType H_12_Block(Derived& section, int uid) {
        return section.hessianWBlockByUID(uid);
      }

      static ListWithCount& edgeSetterComputer12(Derived& section, int i, int j) {
        return section._edgeSetterComputerW[i][j];
      }

    };
    struct EdgeBinaryHessian12_VV {

      using H_12_BlockType = typename Hessian::MatTraitsV::MatrixType::BlockType;

      static H_12_BlockType H_12_Block(Derived& section, int uid) {
        return section.hessianVBlockByUID(uid);
      }

      static ListWithCount & edgeSetterComputer12(Derived& section, int i, int j) {
        return section._edgeSetterComputerV[j][i];
      }

    };

    template<class EdgeBinaryParameter1_T, class EdgeBinaryParameter2_T, class EdgeBinaryHessian12_T>
    struct EdgeBinaryAdapter : public EdgeBinaryParameter1_T, public EdgeBinaryParameter2_T, public EdgeBinaryHessian12_T {
      static constexpr bool isEdgeAcrossSections = !EdgeSameSection<Derived, EdgeBinaryParameter1_T, EdgeBinaryParameter2_T>::value;
    };
    
    template<class EdgeBinaryAdapterT, class EdgeDerived>
    using EdgeBinary = EdgeBinarySectionBaseCRPT<Derived, EdgeBinaryAdapterT, EdgeDerived>;

    using EdgeBinaryAdapterUU = EdgeBinaryAdapter<EdgeBinaryParameter1_U, EdgeBinaryParameter2_U, EdgeBinaryHessian12_UU>;
    using EdgeBinaryAdapterUV = EdgeBinaryAdapter<EdgeBinaryParameter1_U, EdgeBinaryParameter2_V, EdgeBinaryHessian12_UV>;
    using EdgeBinaryAdapterVV = EdgeBinaryAdapter<EdgeBinaryParameter1_V, EdgeBinaryParameter2_V, EdgeBinaryHessian12_VV>;

    template<class EdgeDerived>
    using EdgeBinaryUU = EdgeBinary<EdgeBinaryAdapterUU, EdgeDerived>;
    template<class EdgeDerived>
    using EdgeBinaryUV = EdgeBinary<EdgeBinaryAdapterUV, EdgeDerived>;
    template<class EdgeDerived>
    using EdgeBinaryVV = EdgeBinary<EdgeBinaryAdapterVV, EdgeDerived>;

    template<class EdgeBinaryAdapterT, class EdgeDerived>
    void addEdge(NodeId i, NodeId j, EdgeBinary<EdgeBinaryAdapterT, EdgeDerived>* e);

    //void addEdge(NodeId i, NodeId j, EdgeBinarySectionBase<Derived>* e);
    //void addEdge(int i, int j, int k/*, EdgeTernary* e*/);
    //template<int N>
    //void addEdge(const std::array<int, N>& ids/*, EdgeNAry * e*/);

  };

}
