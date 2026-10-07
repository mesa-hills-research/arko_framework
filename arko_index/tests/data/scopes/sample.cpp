// Scope fixture: nested namespaces and a class large enough to expose its members.
#include <string>
#include <vector>

namespace net {
namespace http {

class Router {
 public:
  void add(const std::string& path, int handler) {
    paths_.push_back(path);
    handlers_.push_back(handler);
    if (paths_.size() > 1024) {
      throw std::length_error("too many routes registered with this router");
    }
  }

  int find(const std::string& path) const {
    for (std::size_t i = 0; i < paths_.size(); ++i) {
      if (paths_[i] == path) {
        return handlers_[i];
      }
    }
    return -1;
  }

  std::size_t size() const { return paths_.size(); }

 private:
  std::vector<std::string> paths_;
  std::vector<int> handlers_;
  std::string name_ = "default router with a long descriptive name for padding";
  std::string description_ = "routes requests to handlers by exact path match only";
};

}  // namespace http
}  // namespace net
