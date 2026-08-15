#include "cloud_quota.h"

#include <stdexcept>

namespace cloud_quota {

quota_bucket::quota_bucket() = default;

void quota_bucket::provision() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (active_ == true) {
        throw std::runtime_error("resource is already active");
    }
    value_ = 0;
    active_ = true;
}

void quota_bucket::grant(int amount) {
    std::unique_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    value_ += amount;
}

void quota_bucket::consume(int amount) {
    std::unique_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > value_) {
        throw std::runtime_error("amount exceeds available value");
    }
    value_ -= amount;
}

void quota_bucket::retire() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    active_ = false;
}

int quota_bucket::available() {
    std::unique_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return value_;
}

}  // namespace cloud_quota
