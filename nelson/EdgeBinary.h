#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSingleSectionBase.h"

namespace nelson {

  class EdgeBinaryBase : public EdgeInterface {
    int _par_1_Id;
    int _par_2_Id;
    int _H_11_uid;
    int _H_12_uid;
    int _H_22_uid;

  protected:
    void setPar_1_Id(int id);
    void setPar_2_Id(int id);
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

    int par_1_Id() const {
      return _par_1_Id;
    }
    int par_2_Id() const {
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

  public:
    EdgeBinarySingleSection();
    virtual ~EdgeBinarySingleSection();


    // virtual void update(bool updateHessians) = 0; // defined in base

  };


}