#include "telemetry_ring.h"

namespace telemetry {

SampleRing::SampleRing(std::size_t depth) : slots_(depth) {
    if (depth == 0) throw std::invalid_argument("depth must be positive");
}

void SampleRing::push(int sample) {
    if (saturated()) throw RingFull();
    slots_.at((head_ + pending_) % slots_.size()) = sample;
    ++pending_;
}

int SampleRing::pop() {
    if (drained()) throw RingEmpty();
    const int sample = slots_.at(head_);
    head_ = (head_ + 1) % slots_.size();
    --pending_;
    return sample;
}

void SampleRing::force(int sample) {
    if (saturated()) {
        head_ = (head_ + 1) % slots_.size();
        --pending_;
    }
    push(sample);
}

void SampleRing::purge() {
    head_ = 0;
    pending_ = 0;
}

std::size_t SampleRing::pending() const { return pending_; }

std::size_t SampleRing::depth() const { return slots_.size(); }

bool SampleRing::saturated() const { return pending_ == slots_.size(); }

bool SampleRing::drained() const { return pending_ == 0; }

}  // namespace telemetry
