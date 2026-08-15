#include "energy_reserve.h"

#include <stdexcept>

namespace energy_reserve {

reserve_meter::reserve_meter() : amount_(0), enabled_(false), mutex_() {}

void reserve_meter::enable() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enabled_ == true) {
        throw std::runtime_error("resource is already active");
    }
    amount_ = 0;
    enabled_ = true;
}

void reserve_meter::add_units(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enabled_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    amount_ += amount;
}

void reserve_meter::consume_units(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enabled_ != true) {
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

void reserve_meter::disable() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enabled_ != true) {
        throw std::runtime_error("resource is not active");
    }
    enabled_ = false;
}

int reserve_meter::remaining_units() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enabled_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return amount_;
}

}  // namespace energy_reserve
