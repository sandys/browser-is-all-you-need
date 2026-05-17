#!/bin/bash
# evaluation.sh - R2E verification script (human + LLM tests)
# Contract:
#   Exit 0: Bug is fixed (tests pass)
#   Exit non-zero: Bug exists (tests fail)
# Timeout: 120 seconds max
# TESTBED=/testbed (repository root)

set -euo pipefail

cd /testbed

echo "=== Applying test patch if present ==="
if [ -f /testbed/patches/test.patch ]; then
    echo "Found test.patch, applying..."
    git apply /testbed/patches/test.patch 2>/dev/null || echo "test.patch already applied or N/A"
fi

echo "=== Installing dependencies ==="
corepack prepare pnpm@latest --activate && CYPRESS_INSTALL_BINARY=0 pnpm install --frozen-lockfile || CYPRESS_INSTALL_BINARY=0 pnpm install

echo "=== Building project ==="
pnpm run build || true

echo "=== Running human-written tests ==="
pnpm test -- packages/react/one-time-password-field/src/one-time-password-field.test.tsx

echo "=== Running LLM verification test ==="
#!/bin/bash

# Install dependencies if needed (though they should already be installed)
# npm install

# Test suite for OneTimePasswordField component
cat <<EOF > test.tsx
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import * as OTP from './packages/react/one-time-password-field/src';
import '@testing-library/jest-dom';

describe('OneTimePasswordField', () => {
  it('should disable all inputs when disabled prop is passed to Root', () => {
    render(
      <OTP.Root disabled>
        <OTP.Input index={0} />
        <OTP.Input index={1} />
        <OTP.Input index={2} />
        <OTP.Input index={3} />
      </OTP.Root>
    );

    const inputs = screen.getAllByRole('textbox');
    inputs.forEach((input) => {
      expect(input).toBeDisabled();
    });
  });

  it('should allow input when disabled prop is not passed to Root', () => {
    render(
      <OTP.Root>
        <OTP.Input index={0} />
        <OTP.Input index={1} />
        <OTP.Input index={2} />
        <OTP.Input index={3} />
      </OTP.Root>
    );

    const inputs = screen.getAllByRole('textbox');
    inputs.forEach((input) => {
      expect(input).not.toBeDisabled();
    });
  });

  it('should disable the hidden input when disabled prop is passed to Root', () => {
    render(
      <OTP.Root disabled>
        <OTP.Input index={0} />
        <OTP.Input index={1} />
        <OTP.Input index={2} />
        <OTP.Input index={3} />
      </OTP.Root>
    );

    const hiddenInput = screen.getByRole('textbox', { hidden: true });
    expect(hiddenInput).toBeDisabled();
  });
});
EOF

# Run the tests using Jest
npx jest --config jest.config.js

echo "All tests passed"
