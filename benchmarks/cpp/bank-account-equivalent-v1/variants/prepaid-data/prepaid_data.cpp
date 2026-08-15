#include "prepaid_data.h"

#include <stdexcept>

namespace prepaid_data {

void data_wallet::connect() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == true) {
        throw std::runtime_error("resource is already active");
    }
    value_ = 0;
    active_ = true;
}

void data_wallet::add_megabytes(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    value_ += amount;
}

void data_wallet::use_megabytes(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
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

void data_wallet::disconnect() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    active_ = false;
}

int data_wallet::remaining_megabytes() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return value_;
}

}  // namespace prepaid_data
