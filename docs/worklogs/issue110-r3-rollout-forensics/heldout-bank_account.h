#pragma once

#include <mutex>
#include <stdexcept>

namespace Bankaccount {

class Bankaccount {
public:
    Bankaccount();
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();

private:
    bool isOpen;
    int balance;
    std::mutex mtx;
};

} // namespace Bankaccount
