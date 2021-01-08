#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSingleSectionBase.h"

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
  class EdgeBinarySingleSection : public EdgeBinaryBase, public EdgeSingleSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;

    class HessianUpdater_11 : public EdgeHessianUpdater {
      EdgeBinarySingleSection* _e;
    public:
      HessianUpdater_11(EdgeBinarySingleSection* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_11();
      }
    };
    class HessianUpdater_12 : public EdgeHessianUpdater {
      EdgeBinarySingleSection* _e;
    public:
      HessianUpdater_12(EdgeBinarySingleSection* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_12();
      }
    };
    class HessianUpdater_22 : public EdgeHessianUpdater {
      EdgeBinarySingleSection* _e;
    public:
      HessianUpdater_22(EdgeBinarySingleSection* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH_22();
      }
    };
  public:
    EdgeBinarySingleSection();
    virtual ~EdgeBinarySingleSection();

    virtual void updateH_11() = 0;
    virtual void updateH_12() = 0;
    virtual void updateH_22() = 0;

    // virtual void update(bool updateHessians) = 0; // defined in base

  };

  template<class Section, class Derived>
  class EdgeBinarySingleSectionCRPT : public EdgeBinarySingleSection<Section> {

  public:

    const typename Section::ParameterType& parameter_1() const {
      return this->section().parameter(this->par_1_Id());
    }
    const typename Section::ParameterType& parameter_2() const {
      return this->section().parameter(this->par_2_Id());
    }

    void updateH_11() override final {
      assert(this->H_11_Uid() >= 0);
      assert(this->par_1_Id().isVariable());

      typename Section::Hessian::MatTraits::MatrixType::BlockType bH = this->section().hessianBlockByUID(this->H_11_Uid());
      typename Section::Hessian::VecType::SegmentType bV = this->section().bVectorSegment(this->par_1_Id().id());
      static_cast<Derived*>(this)->updateH11Block(bH, bV);
    }
    void updateH_12() override final {
      assert(this->H_12_Uid() >= 0);
      typename Section::Hessian::MatTraits::MatrixType::BlockType bH = this->section().hessianBlockByUID(this->H_12_Uid());
      static_cast<Derived*>(this)->updateH12Block(bH);
    }
    void updateH_22() override final {
      assert(this->H_22_Uid() >= 0);
      assert(this->par_2_Id().isVariable());

      typename Section::Hessian::MatTraits::MatrixType::BlockType bH = this->section().hessianBlockByUID(this->H_22_Uid());
      typename Section::Hessian::VecType::SegmentType bV = this->section().bVectorSegment(this->par_2_Id().id());
      static_cast<Derived*>(this)->updateH22Block(bH, bV);
    }
  };

}