#include "library_credit.h"

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
        library_credit::patron_account subject{};
        subject.enroll();
        checks.expect(subject.credit() == 0, "newly_started_zero");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        checks.expect(subject.credit() == 100, "single_credit");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        subject.add_credit(50);
        checks.expect(subject.credit() == 150, "multiple_credits");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        subject.use_credit(75);
        checks.expect(subject.credit() == 25, "single_debit");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        subject.use_credit(80);
        subject.use_credit(20);
        checks.expect(subject.credit() == 0, "multiple_debits");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        subject.add_credit(110);
        subject.use_credit(200);
        subject.add_credit(60);
        subject.use_credit(50);
        checks.expect(subject.credit() == 20, "sequential_operations");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.close();
        checks.expect_runtime_error(
            [&]() { (void)subject.credit(); },
            "value_after_stop_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.close();
        checks.expect_runtime_error(
            [&]() { subject.add_credit(50); },
            "credit_after_stop_throws");
    }
    {
        library_credit::patron_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.add_credit(50); },
            "credit_before_start_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.close();
        checks.expect_runtime_error(
            [&]() { subject.use_credit(50); },
            "debit_after_stop_throws");
    }
    {
        library_credit::patron_account subject{};
        checks.expect_runtime_error(
            [&]() { subject.close(); },
            "stop_before_start_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        checks.expect_runtime_error(
            [&]() { subject.enroll(); },
            "start_twice_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(50);
        subject.close();
        subject.enroll();
        checks.expect(subject.credit() == 0, "restart_resets_zero");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(25);
        checks.expect_runtime_error(
            [&]() { subject.use_credit(50); },
            "overdraft_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        subject.add_credit(100);
        checks.expect_runtime_error(
            [&]() { subject.use_credit(-50); },
            "negative_debit_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        checks.expect_runtime_error(
            [&]() { subject.add_credit(-50); },
            "negative_credit_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.add_credit(1);
                std::this_thread::sleep_for(5ms);
                subject.use_credit(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.credit() == 0, "concurrent_transactions");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        checks.expect_runtime_error(
            [&]() { subject.use_credit(0); },
            "zero_debit_throws");
    }
    {
        library_credit::patron_account subject{};
        subject.enroll();
        checks.expect_runtime_error(
            [&]() { subject.add_credit(0); },
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
