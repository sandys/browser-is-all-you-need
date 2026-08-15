#include "library_credit.h"

#include <stdexcept>

namespace library_credit {

patron_account::patron_account() = default;

void patron_account::enroll() {
    std::scoped_lock<std::mutex> guard(mutex_);
    if (active_ == true) {
        throw std::runtime_error("resource is already active");
    }
    value_ = 0;
    active_ = true;
}

void patron_account::add_credit(int amount) {
    std::scoped_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    value_ += amount;
}

void patron_account::use_credit(int amount) {
    std::scoped_lock<std::mutex> guard(mutex_);
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

void patron_account::close() {
    std::scoped_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    active_ = false;
}

int patron_account::credit() {
    std::scoped_lock<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return value_;
}

}  // namespace library_credit
