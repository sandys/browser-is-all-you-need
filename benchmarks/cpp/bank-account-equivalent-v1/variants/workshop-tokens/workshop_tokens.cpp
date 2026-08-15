#include "workshop_tokens.h"

#include <stdexcept>

namespace workshop_tokens {

token_box::token_box() : count_(0), unlocked_(false) {}

void token_box::unlock() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (unlocked_ == true) {
        throw std::runtime_error("resource is already active");
    }
    count_ = 0;
    unlocked_ = true;
}

void token_box::add_tokens(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (unlocked_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    count_ += amount;
}

void token_box::take_tokens(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (unlocked_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > count_) {
        throw std::runtime_error("amount exceeds available value");
    }
    count_ -= amount;
}

void token_box::lock() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (unlocked_ != true) {
        throw std::runtime_error("resource is not active");
    }
    unlocked_ = false;
}

int token_box::token_count() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (unlocked_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return count_;
}

}  // namespace workshop_tokens
