#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSectionBase.h"

#include <array>
#include <vector>

#include "mat/Global.h"

#include <Eigen/Core>

#include "EdgeNary.h"

namespace nelson {

  template<int N1, int N2>
  class EdgeBNaryContainer {
  protected:
    EdgeNaryContainer<N1> _par1;
    EdgeNaryContainer<N2> _par2;

  public:
    EdgeBNaryContainer(int n1, int n2) : _par1(n1), _par2(n2) {}

    EdgeNaryContainer<N1>& par1() {
      return _par1;
    }
    EdgeNaryContainer<N2>& par2() {
      return _par2;
    }

    const EdgeNaryContainer<N1>& par1() const {
      return _par1;
    }
    const EdgeNaryContainer<N2>& par2() const {
      return _par2;
    }

    int numParams1() const { return _par1.numParams(); }
    int numParams2() const { return _par2.numParams(); }


  };

  template<int N1, int N2>
  class EdgeBNaryBase : public EdgeInterface {
    EdgeBNaryContainer<N1, N2> _parIds;
    Eigen::Matrix<int, N1, N1> _HU_uid; //note, only triag up used.... can we do better storing a linear vector only?
    Eigen::Matrix<int, N1, N2> _HW_uid; 
    Eigen::Matrix<int, N2, N2> _HV_uid; //note, only triag up used.... can we do better storing a linear vector only?

  protected:
    void setPar1Id(int i, NodeId id);
    void setPar2Id(int i, NodeId id);

    void setH_U_Uid(int i, int j, int uid);
    void setH_W_Uid(int i, int j, int uid);
    void setH_V_Uid(int i, int j, int uid);

    class EdgeUID_U_Setter final : public EdgeUIDSetterInterface {
      EdgeBNaryBase* _e;
      int _i, _j;
    public:
      EdgeUID_U_Setter(EdgeBNaryBase* e, int i, int j) : _e(e), _i(i), _j(j) {}

      void setUID(int uid) override;

    };
    class EdgeUID_W_Setter final : public EdgeUIDSetterInterface {
      EdgeBNaryBase* _e;
      int _i, _j;
    public:
      EdgeUID_W_Setter(EdgeBNaryBase* e, int i, int j) : _e(e), _i(i), _j(j) {}

      void setUID(int uid) override;

    };
    class EdgeUID_V_Setter final : public EdgeUIDSetterInterface {
      EdgeBNaryBase* _e;
      int _i, _j;
    public:
      EdgeUID_V_Setter(EdgeBNaryBase* e, int i, int j) : _e(e), _i(i), _j(j) {}

      void setUID(int uid) override;

    };

  public:

    // TODO: better if 2 constructor with a enable_if like mechanism....
    EdgeBNaryBase(int n1 = N1, int n2 = N2);
    virtual ~EdgeBNaryBase();

    NodeId par_1_Id(int i) const {
      return this->_parIds.par1().parId()[i];
    }
    NodeId par_2_Id(int i) const {
      return this->_parIds.par2().parId()[i];
    }

    int HUid_U(int i, int j) const {
      assert(j >= i);
      return _HU_uid(i, j);
    }
    int HUid_V(int i, int j) const {
      assert(j >= i);
      return _HV_uid(i, j);
    }
    int HUid_W(int i, int j) const {
      return _HW_uid(i, j);
    }

    int numParams1() const {
      return _parIds.numParams1();
    }
    int numParams2() const {
      return _parIds.numParams2();
    }

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section, int N1, int N2>
  class EdgeBNarySectionBase : public EdgeBNaryBase<N1,N2>, public EdgeSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;
    template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv> friend class DoubleSection;

    class HessianUpdater_U : public EdgeHessianUpdater {
      EdgeBNarySectionBase* _e;
      int _i, _j;
    public:
      HessianUpdater_U(EdgeBNarySectionBase* e, int i, int j) : _e(e), _i(i), _j(j) {

      }

      void updateH(bool transpose) override {
        _e->updateH_U(_i, _j, transpose);
      }
    };
    class HessianUpdater_V : public EdgeHessianUpdater {
      EdgeBNarySectionBase* _e;
      int _i, _j;
    public:
      HessianUpdater_V(EdgeBNarySectionBase* e, int i, int j) : _e(e), _i(i), _j(j) {

      }

      void updateH(bool transpose) override {
        _e->updateH_V(_i, _j, transpose);
      }
    };
    class HessianUpdater_W : public EdgeHessianUpdater {
      EdgeBNarySectionBase* _e;
      int _i, _j;
    public:
      HessianUpdater_W(EdgeBNarySectionBase* e, int i, int j) : _e(e), _i(i), _j(j) {

      }

      void updateH(bool transpose) override {
        _e->updateH_W(_i, _j, transpose);
      }
    };

  public:

    EdgeBNarySectionBase(int size1 = N1, int size2 = N2);
    virtual ~EdgeBNarySectionBase();

    // virtual void update(bool updateHessians) = 0; // defined in base

    virtual void updateH_U(int i, int j, bool transpose) = 0;
    virtual void updateH_V(int i, int j, bool transpose) = 0;
    virtual void updateH_W(int i, int j, bool transpose) = 0;

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section, int N1, int N2, class SectionAdapter, class Derived>
  class EdgeBNarySectionBaseCRPT : public EdgeBNarySectionBase<Section, N1, N2> {

  public:
    EdgeBNarySectionBaseCRPT(int size1 = N1, int size2 = N2);
    virtual ~EdgeBNarySectionBaseCRPT();

    inline const typename SectionAdapter::Parameter_1_Type& parameter_1(int i) const {
      return SectionAdapter::parameter1(this->section(), this->par_1_Id(i));
    }
    inline const typename SectionAdapter::Parameter_2_Type& parameter_2(int i) const {
      return SectionAdapter::parameter2(this->section(), this->par_2_Id(i));
    }

    void updateH_U(int i, int j, bool transpose) override final {
      assert(this->HUid_U(i, j) >= 0);
      assert(this->par_1_Id(i).isVariable());
      assert(this->par_1_Id(j).isVariable());

      typename SectionAdapter::H_11_BlockType  bH = SectionAdapter::H_11_Block(this->section(), this->HUid_U(i, j));
      if (i == j) {
        assert(transpose == false);
        typename SectionAdapter::BSegmentType bV = SectionAdapter::b_1_Segment(this->section(), this->par_1_Id(i).id());
        static_cast<Derived*>(this)->updateHUBlock(i, bH, bV);
      }
      else {
        static_cast<Derived*>(this)->updateHUBlock(i, j, bH, transpose);
      }
    }
    void updateH_V(int i, int j, bool transpose) override final {
      assert(this->HVid_V(i, j) >= 0);
      assert(this->par_2_Id(i).isVariable());
      assert(this->par_2_Id(j).isVariable());

      typename SectionAdapter::H_22_BlockType  bH = SectionAdapter::H_22_Block(this->section(), this->HVid_U(i, j));
      if (i == j) {
        assert(transpose == false);
        typename SectionAdapter::BSegmentType bV = SectionAdapter::b_2_Segment(this->section(), this->par_2_Id(i).id());
        static_cast<Derived*>(this)->updateHVBlock(i, bH, bV);
      }
      else {
        static_cast<Derived*>(this)->updateHVBlock(i, j, bH, transpose);
      }
    }
    void updateH_W(int i, int j, bool transpose) override final {
      assert(this->HWid_V(i, j) >= 0);
      assert(this->par_1_Id(i).isVariable());
      assert(this->par_2_Id(j).isVariable());

      typename SectionAdapter::H_11_BlockType  bH = SectionAdapter::H_12_Block(this->section(), this->HVid_U(i, j));
      static_cast<Derived*>(this)->updateHVBlock(i, j, bH, transpose);
    }

  };
}