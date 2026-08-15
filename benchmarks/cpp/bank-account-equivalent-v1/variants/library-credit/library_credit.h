#pragma once

#include <mutex>

namespace library_credit {

class patron_account {
public:
    patron_account();
    void enroll();
    void add_credit(int amount);
    void use_credit(int amount);
    void close();
    int credit();

private:
    int value_{0};
    bool active_{false};
    std::mutex mutex_{};
};

}  // namespace library_credit
