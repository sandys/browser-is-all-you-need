#!/bin/bash
# evaluation.sh - R2E verification script (LLM-generated tests)
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

echo "=== Running LLM verification test ==="
#!/bin/bash

cat <<EOF > packages/react/test/heading.test.tsx
import { render } from '@testing-library/react';
import { Heading } from '@chakra-ui/react';
import { HeadingWithGradient } from 'compositions/src/examples/heading-with-gradient';

describe('Heading with Gradient', () => {
  it('should apply background-clip: text style when bgClip="text" is used with gradient', () => {
    const { container } = render(<HeadingWithGradient />);
    const headingElement = container.firstChild as HTMLElement;

    expect(headingElement).toHaveStyle('background-clip: text');
    expect(headingElement).toHaveStyle('color: transparent');
    expect(headingElement).toHaveStyle('background-image: linear-gradient(to left, var(--chakra-colors-red-500), var(--chakra-colors-blue-500))');
  });

  it('should render the heading with the correct text', () => {
    const { getByText } = render(<HeadingWithGradient />);
    const headingText = getByText('The quick brown fox jumps over the lazy dog');
    expect(headingText).toBeInTheDocument();
  });
});
EOF

yarn test packages/react/test/heading.test.tsx

echo "All tests passed"
