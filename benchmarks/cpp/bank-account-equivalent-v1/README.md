# Bank-account-equivalent C++ benchmark variants

This directory contains ten independently named benchmark fixtures for
the complete fixed26 `bank-account` state-machine, exception, file-layout,
and concurrency contract. They are intended only for new benchmark
construction and robustness evaluation. They are explicitly excluded
from SFT, RL, distillation, and every other training corpus.

## Coverage and lineage

The parent static test SHA-256 is
`3696b9383f62ab639ad0a26610410fb662b6b927f2fe1dee956b849ff8dcf5c8`.
Each variant maps all 17 parent cases one-for-one and adds two tests for
zero-valued credit and debit operations. Those two checks close the
historical gap where the prompt said “non-positive” but the parent suite
exercised only negative values.

Every variant therefore contains 19 assertions:

- six ordinary sequential state/value cases;
- seven lifecycle and inactive-state exception cases;
- five amount-boundary cases, including negative, zero, and overdraft;
- one 1,000-thread transaction test.

The historical July 27 prompt omitted reset-on-reopen even though the
parent test required it. These prompts include the corrected lifecycle
rule. They also state default construction and header/source ownership
explicitly because six of the eight observed failures were C++ delivery
failures rather than incorrect account arithmetic.

| Variant | Public class | Operations |
| --- | --- | --- |
| [Secure wallet](variants/secure-wallet/PROMPT.md) | `secure_wallet::wallet_account` | `activate` / `deposit` / `withdraw` / `deactivate` / `balance` |
| [Energy reserve](variants/energy-reserve/PROMPT.md) | `energy_reserve::reserve_meter` | `enable` / `add_units` / `consume_units` / `disable` / `remaining_units` |
| [Arcade card](variants/arcade-card/PROMPT.md) | `arcade_card::player_card` | `issue` / `load` / `spend` / `revoke` / `credits` |
| [Inventory ledger](variants/inventory-ledger/PROMPT.md) | `inventory_ledger::stock_ledger` | `begin` / `receive` / `dispatch` / `end` / `quantity` |
| [Transit pass](variants/transit-pass/PROMPT.md) | `transit_pass::fare_pass` | `activate` / `top_up` / `charge` / `suspend` / `funds` |
| [Cloud quota](variants/cloud-quota/PROMPT.md) | `cloud_quota::quota_bucket` | `provision` / `grant` / `consume` / `retire` / `available` |
| [Library credit](variants/library-credit/PROMPT.md) | `library_credit::patron_account` | `enroll` / `add_credit` / `use_credit` / `close` / `credit` |
| [Reward points](variants/reward-points/PROMPT.md) | `reward_points::reward_account` | `open` / `earn` / `redeem` / `close` / `points` |
| [Prepaid data](variants/prepaid-data/PROMPT.md) | `prepaid_data::data_wallet` | `connect` / `add_megabytes` / `use_megabytes` / `disconnect` / `remaining_megabytes` |
| [Workshop tokens](variants/workshop-tokens/PROMPT.md) | `workshop_tokens::token_box` | `unlock` / `add_tokens` / `take_tokens` / `lock` / `token_count` |

## Verify

From the repository root:

```bash
python3 benchmarks/cpp/bank-account-equivalent-v1/verify.py
```

Verification checks lineage and file hashes, exact API and prompt
coverage, the 17+2 test inventory, standalone header compilation, strict
C++17 compilation, a deadlock timeout, and all 190 runtime assertions.
The committed
[`verification_receipt.json`](verification_receipt.json) records the
successful environment and per-variant results.

The deterministic source generator is
[`scripts/build_bank_account_equivalent_benchmarks.py`](../../../scripts/build_bank_account_equivalent_benchmarks.py).
