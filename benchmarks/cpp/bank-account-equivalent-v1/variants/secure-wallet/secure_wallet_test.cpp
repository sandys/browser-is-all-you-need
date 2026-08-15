#include "secure_wallet.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class check_suite {
public:
    void expect(bool condition, std::string_view description) {
        ++count_;
        if (!condition) {
            failed_ = true;
            std::cerr << "FAILED: " << description << '\n';
        }
    }

    template <typename Function>
    void expect_runtime_error(Function&& function, std::string_view description) {
        ++count_;
        try {
            std::forward<Function>(function)();
        } catch (std::runtime_error const&) {
            return;
        } catch (std::exception const& error) {
            failed_ = true;
            std::cerr << "FAILED: " << description
                      << " threw a different exception: " << error.what() << '\n';
            return;
        }
        failed_ = true;
        std::cerr << "FAILED: " << description
                  << " did not throw std::runtime_error\n";
    }

    std::size_t count() const { return count_; }
    bool failed() const { return failed_; }

private:
    std::size_t count_{0};
    bool failed_{false};
};

}  // namespace

int main() {
    check_suite checks;

    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        checks.expect(subject.balance() == 0, "newly_started_zero");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        checks.expect(subject.balance() == 100, "single_credit");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        subject.deposit(50);
        checks.expect(subject.balance() == 150, "multiple_credits");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        subject.withdraw(75);
        checks.expect(subject.balance() == 25, "single_debit");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        subject.withdraw(80);
        subject.withdraw(20);
        checks.expect(subject.balance() == 0, "multiple_debits");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        subject.deposit(110);
        subject.withdraw(200);
        subject.deposit(60);
        subject.withdraw(50);
        checks.expect(subject.balance() == 20, "sequential_operations");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deactivate();
        checks.expect_runtime_error(
            [&]() { (void)subject.balance(); },
            "value_after_stop_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deactivate();
        checks.expect_runtime_error(
            [&]() { subject.deposit(50); },
            "credit_after_stop_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.deposit(50); },
            "credit_before_start_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deactivate();
        checks.expect_runtime_error(
            [&]() { subject.withdraw(50); },
            "debit_after_stop_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.deactivate(); },
            "stop_before_start_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.activate(); },
            "start_twice_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(50);
        subject.deactivate();
        subject.activate();
        checks.expect(subject.balance() == 0, "restart_resets_zero");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(25);
        checks.expect_runtime_error(
            [&]() { subject.withdraw(50); },
            "overdraft_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        subject.deposit(100);
        checks.expect_runtime_error(
            [&]() { subject.withdraw(-50); },
            "negative_debit_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.deposit(-50); },
            "negative_credit_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.deposit(1);
                std::this_thread::sleep_for(5ms);
                subject.withdraw(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.balance() == 0, "concurrent_transactions");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.withdraw(0); },
            "zero_debit_throws");
    }
    {
        secure_wallet::wallet_account subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.deposit(0); },
            "zero_credit_throws");
    }

    if (checks.failed() || checks.count() != 19U) {
        std::cerr << "Verification failed after " << checks.count()
                  << " assertions\n";
        return 1;
    }

    std::cout << "All tests passed (19 assertions in 19 test cases)\n";
    return 0;
}
