#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace telemetry {

class RingEmpty : public std::logic_error {
  public:
    RingEmpty() : std::logic_error("ring holds no samples") {}
};

class RingFull : public std::logic_error {
  public:
    RingFull() : std::logic_error("ring is saturated") {}
};

class SampleRing {
  public:
    explicit SampleRing(std::size_t depth);

    void push(int sample);
    int pop();
    void force(int sample);
    void purge();

    std::size_t pending() const;
    std::size_t depth() const;
    bool saturated() const;
    bool drained() const;

  private:
    std::vector<int> slots_;
    std::size_t head_{0};
    std::size_t pending_{0};
};

}  // namespace telemetry
