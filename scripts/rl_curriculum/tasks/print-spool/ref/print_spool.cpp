#include "print_spool.h"

namespace spool {

PrintSpool::PrintSpool(std::size_t slots) : ring_(slots) {
    if (slots == 0) throw std::invalid_argument("slots must be positive");
}

void PrintSpool::queue(const std::string& job) {
    if (full()) throw SpoolFull();
    ring_.at((front_ + backlog_) % ring_.size()) = job;
    ++backlog_;
}

std::string PrintSpool::release() {
    if (idle()) throw SpoolEmpty();
    const std::string job = ring_.at(front_);
    front_ = (front_ + 1) % ring_.size();
    --backlog_;
    return job;
}

void PrintSpool::displace(const std::string& job) {
    if (full()) {
        front_ = (front_ + 1) % ring_.size();
        --backlog_;
    }
    queue(job);
}

void PrintSpool::abandon() {
    front_ = 0;
    backlog_ = 0;
}

std::size_t PrintSpool::backlog() const { return backlog_; }

std::size_t PrintSpool::slots() const { return ring_.size(); }

bool PrintSpool::full() const { return backlog_ == ring_.size(); }

bool PrintSpool::idle() const { return backlog_ == 0; }

}  // namespace spool
