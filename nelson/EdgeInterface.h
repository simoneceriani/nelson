#pragma once

namespace nelson {

  class EdgeInterface {

    double _chi2;

  protected:
    inline void setChi2(double v) {
      _chi2 = v;
    }
  public:

    EdgeInterface();
    virtual ~EdgeInterface();
    virtual void update(bool updateHessians) = 0;

    inline double chi2() const {
      return _chi2;
    }

  };

  class EdgeUIDSetterInterface {
  public:
    virtual void setUID(int uid) = 0;
  };

  class EdgeHessianUpdater {
  public:
    virtual void updateH() = 0;
  };

  enum class NodeType {
    Variable,
    Fixed
  };

  class NodeId {
    int _id;
    NodeType _type;

  public:
    NodeId(int id, NodeType nt = NodeType::Variable) :_id(id), _type(nt) {}

    int id() const { return _id; };
    NodeType type() const { return _type; }
    bool isVariable() const { return _type == NodeType::Variable; }
    bool isFixed() const { return _type == NodeType::Fixed; }

    static NodeId fixed(int i) {
      return NodeId(i, NodeType::Fixed);
    }
  };

}