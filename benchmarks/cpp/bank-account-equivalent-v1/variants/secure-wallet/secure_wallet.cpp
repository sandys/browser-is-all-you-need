#include "secure_wallet.h"

#include <stdexcept>

namespace secure_wallet {

wallet_account::wallet_account() = default;

void wallet_account::activate() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == true) {
        throw std::runtime_error("resource is already active");
    }
    value_ = 0;
    active_ = true;
}

void wallet_account::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    value_ += amount;
}

void wallet_account::withdraw(int amount) {
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

void wallet_account::deactivate() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    active_ = false;
}

int wallet_account::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return value_;
}

}  // namespace secure_wallet
