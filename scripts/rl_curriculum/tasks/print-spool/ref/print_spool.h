#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace spool {

class SpoolEmpty : public std::logic_error {
  public:
    SpoolEmpty() : std::logic_error("spool has no waiting jobs") {}
};

class SpoolFull : public std::logic_error {
  public:
    SpoolFull() : std::logic_error("every spool slot is taken") {}
};

class PrintSpool {
  public:
    explicit PrintSpool(std::size_t slots);

    void queue(const std::string& job);
    std::string release();
    void displace(const std::string& job);
    void abandon();

    std::size_t backlog() const;
    std::size_t slots() const;
    bool full() const;
    bool idle() const;

  private:
    std::vector<std::string> ring_;
    std::size_t front_{0};
    std::size_t backlog_{0};
};

}  // namespace spool
