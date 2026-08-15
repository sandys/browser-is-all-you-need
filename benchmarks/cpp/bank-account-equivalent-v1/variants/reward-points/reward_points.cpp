#include "reward_points.h"

#include <stdexcept>

namespace reward_points {

reward_account::reward_account() = default;

void reward_account::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_.active == true) {
        throw std::runtime_error("resource is already active");
    }
    state_.value = 0;
    state_.active = true;
}

void reward_account::earn(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_.active != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    state_.value += amount;
}

void reward_account::redeem(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_.active != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > state_.value) {
        throw std::runtime_error("amount exceeds available value");
    }
    state_.value -= amount;
}

void reward_account::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_.active != true) {
        throw std::runtime_error("resource is not active");
    }
    state_.active = false;
}

int reward_account::points() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (state_.active != true) {
        throw std::runtime_error("resource is not active");
    }
    return state_.value;
}

}  // namespace reward_points
