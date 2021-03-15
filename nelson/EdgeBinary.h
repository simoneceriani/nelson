#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSectionBase.h"

namespace nelson {

  class EdgeBinaryBase : public EdgeInterface {
    NodeId _par_1_Id;
    NodeId _par_2_Id;
    int _H_11_uid;
    int _H_12_uid;
    int _H_22_uid;

  protected:
    void setPar_1_Id(NodeId id);
    void setPar_2_Id(NodeId id);
    void setH_11_Uid(int uid);
    void setH_12_Uid(int uid);
    void setH_22_Uid(int uid);

    class EdgeUID_11_Setter final : public EdgeUIDSetterInterface {
      EdgeBinaryBase* _e;
    public:
      EdgeUID_11_Setter(EdgeBinaryBase* e) : _e(e) {}

      void setUID(int uid) override;
    };
    class EdgeUID_12_Setter final : public EdgeUIDSetterInterface {
      EdgeBinaryBase* _e;
    public:
      EdgeUID_12_Setter(EdgeBinaryBase* e) : _e(e) {}

      void setUID(int uid) override;
    };
    class EdgeUID_22_Setter final : public EdgeUIDSetterInterface {
      EdgeBinaryBase* _e;
    public:
      EdgeUID_22_Setter(EdgeBinaryBase* e) : _e(e) {}

      void setUID(int uid) override;
    };

  public:
    EdgeBinaryBase();
    virtual ~EdgeBinaryBase();

    NodeId par_1_Id() const {
      return _par_1_Id;
    }
    NodeId par_2_Id() const {
      return _par_2_Id;
    }

    int H_11_Uid() const {
      return _H_11_uid;
    }
    int H_12_Uid() const {
      return _H_12_uid;
    }
    int H_22_Uid() const {
      return _H_22_uid;
    }

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section>
  class EdgeBinarySectionBase : public EdgeBinaryBase, public EdgeSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;
    template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv> friend class DoubleSection;

    class HessianUpdater_11 : public EdgeHessianUpdater {
      EdgeBinarySectionBase* _e;
    public:
      HessianUpdater_11(EdgeBinarySectionBase* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_11();
      }
    };
    class HessianUpdater_12 : public EdgeHessianUpdater {
      EdgeBinarySectionBase* _e;
    public:
      HessianUpdater_12(EdgeBinarySectionBase* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_12();
      }
    };
    class HessianUpdater_22 : public EdgeHessianUpdater {
      EdgeBinarySectionBase* _e;
    public:
      HessianUpdater_22(EdgeBinarySectionBase* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_22();
      }
    };
  public:
    EdgeBinarySectionBase();
    virtual ~EdgeBinarySectionBase();

    virtual void updateH_11() = 0;
    virtual void updateH_12() = 0;
    virtual void updateH_22() = 0;

    // virtual void update(bool updateHessians) = 0; // defined in base

  };

  template<class Section, class SectionAdapter, class Derived>
  class EdgeBinarySectionBaseCRPT : public EdgeBinarySectionBase<Section> {

  public:

    inline const typename SectionAdapter::Parameter_1_Type& parameter_1() const {
      return SectionAdapter::parameter1(this->section(), this->par_1_Id());
    }
    inline const typename SectionAdapter::Parameter_2_Type& parameter_2() const {
      return SectionAdapter::parameter2(this->section(), this->par_2_Id());
    }

    void updateH_11() override final {
      assert(this->H_11_Uid() >= 0);
      assert(this->par_1_Id().isVariable());

      typename SectionAdapter::H_11_BlockType  bH = SectionAdapter::H_11_Block(this->section(), this->H_11_Uid());
      typename SectionAdapter::B_1_SegmentType bV = SectionAdapter::b_1_Segment(this->section(), this->par_1_Id().id());
      static_cast<Derived*>(this)->updateH11Block(bH, bV);
    }
    void updateH_12() override final {
      assert(this->H_12_Uid() >= 0);
      typename SectionAdapter::H_12_BlockType bH = SectionAdapter::H_12_Block(this->section(), this->H_12_Uid());
      static_cast<Derived*>(this)->updateH12Block(bH);
    }
    void updateH_22() override final {
      assert(this->H_22_Uid() >= 0);
      assert(this->par_2_Id().isVariable());

      typename SectionAdapter::H_22_BlockType  bH = SectionAdapter::H_22_Block(this->section(), this->H_22_Uid());
      typename SectionAdapter::B_2_SegmentType bV = SectionAdapter::b_2_Segment(this->section(), this->par_2_Id().id());
      static_cast<Derived*>(this)->updateH22Block(bH, bV);
    }
  };

}