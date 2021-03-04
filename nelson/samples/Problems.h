#include <vector>
#include <string>

struct Problem {
  int nCameras;
  int nPoints;
  std::vector<std::pair<int, int>> edges;

  bool load(const std::string& file);
};

struct ProblemCollections {
  static const Problem ladybug_49_7776;
};

