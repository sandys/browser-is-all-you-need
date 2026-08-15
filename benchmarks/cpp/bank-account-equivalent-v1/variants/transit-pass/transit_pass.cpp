#include "transit_pass.h"

#include <stdexcept>

namespace transit_pass {

fare_pass::fare_pass() = default;

void fare_pass::activate() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (status_ == status::active) {
        throw std::runtime_error("resource is already active");
    }
    amount_ = 0;
    status_ = status::active;
}

void fare_pass::top_up(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (status_ != status::active) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    amount_ += amount;
}

void fare_pass::charge(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (status_ != status::active) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > amount_) {
        throw std::runtime_error("amount exceeds available value");
    }
    amount_ -= amount;
}

void fare_pass::suspend() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (status_ != status::active) {
        throw std::runtime_error("resource is not active");
    }
    status_ = status::inactive;
}

int fare_pass::funds() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (status_ != status::active) {
        throw std::runtime_error("resource is not active");
    }
    return amount_;
}

}  // namespace transit_pass
