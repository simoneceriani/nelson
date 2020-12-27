#pragma once
#include "Global.h"

#include "EdgeInterface.h"

namespace nelson {

  class EdgeBinary : public EdgeInterface {

    template<class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;

    int _par_1_Id;
    int _par_2_Id;
    int _H_11_uid;
    int _H_12_uid;
    int _H_22_uid;

    void setPar_1_Id(int id);
    void setPar_2_Id(int id);
    void setH_11_Uid(int uid);
    void setH_12_Uid(int uid);
    void setH_22_Uid(int uid);

    class EdgeUID_11_Setter final : public EdgeUIDSetterInterface {
      EdgeBinary* _e;
    public:
      EdgeUID_11_Setter(EdgeBinary* e) : _e(e) {}

      void setUID(int uid) override;
    };
    class EdgeUID_12_Setter final : public EdgeUIDSetterInterface {
      EdgeBinary* _e;
    public:
      EdgeUID_12_Setter(EdgeBinary* e) : _e(e) {}

      void setUID(int uid) override;
    };
    class EdgeUID_22_Setter final : public EdgeUIDSetterInterface {
      EdgeBinary* _e;
    public:
      EdgeUID_22_Setter(EdgeBinary* e) : _e(e) {}

      void setUID(int uid) override;
    };

  public:
    EdgeBinary();
    virtual ~EdgeBinary();


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

    virtual void update(bool updateHessians) = 0;

  };


}