#pragma once

#include <mutex>

namespace secure_wallet {

class wallet_account {
public:
    wallet_account();
    void activate();
    void deposit(int amount);
    void withdraw(int amount);
    void deactivate();
    int balance();

private:
    int value_{0};
    bool active_{false};
    std::mutex mutex_{};
};

}  // namespace secure_wallet
