<!-- harbor_instruction_version: 1 -->
# radix-ui__primitives-3548

**Repository:** radix-ui/primitives

## Task Summary

disabled prop on RadixOTPField.Root doesn’t disable the child inputs

## Expected Behavior

# disabled prop on RadixOTPField.Root doesn’t disable the child inputs

## Issue Description
## Bug report

### Current Behavior

When passing the disabled prop to the One-Time-Password field components, the inputs remain interactive—they can still be focused and edited—and styling never reflects a disabled state. Even if you wrap `<RadixOTPField.Root disabled={true}>`, neither the individual <Input> nor the hidden input respects or applies that flag.

### Expected behavior

A consumer should be able to disable the entire OTP field by doing:
`<RadixOTPField.Root disabled />`

- All digit inputs should become non-focusable and non-editable.
- The hidden input should also be marked disabled.
- The component should emit native disabled attributes and allow CSS selectors (e.g. [disabled], .disabled) to style the root and inputs.

### Reproducible example

[CodeSandbox Template](https://codesandbox.io/s/2r30e)

- Pass disabled to <RadixOTPField.Root>.
- Notice you can still click into and type in each digit <Input>.
- Inspect the DOM: the inputs never receive disabled="true".

### Suggested solution

Have the package forward an explicit disabled prop to each child:

And in Input.tsx and HiddenInput.tsx, ensure the <input> element accepts and applies disabled={disabled}.

### Additional context

<!-- Add any other context about the problem here.  -->

### Your environment

<!-- Very important for us to help you debug. Please fill this out! -->

| Software         | Name(s) | Version |
| ---------------- | ------- | ------- |
| Radix Package(s) |   @radix-ui/react-one-time-password-field | ^1.0.0 |
| React            | n/a     |         | 18.2.0
| Browser          |         |         |
| Assistive tech   |         |         |
| Node             | n/a     |         |
| npm/yarn/pnpm    |         |         |
| Operating System |         |         |

## Fix: fix: [One-Time Password Field] disable all inputs when disable is true

## Additional Context
### Description
close #3545

- Add test for `disable`
- disable all inputs when `disable` is true.

## Likely Fix Direction

Have the package forward an explicit disabled prop to each child:

And in Input.tsx and HiddenInput.tsx, ensure the <input> element accepts and applies disabled={disabled}.

## Likely Touch Points

- `.changeset/kind-icons-tan.md`
- `packages/react/one-time-password-field/src/one-time-password-field.test.tsx`
- `packages/react/one-time-password-field/src/one-time-password-field.tsx`

## Success Criteria

- Fix the bug described above without breaking the existing preview.
- Ensure the rendered preview at `/iframe.html?id=components-onetimepasswordfield--uncontrolled&viewMode=story&args=disabled:true` shows the intended behavior.
- Keep the repository build and tests passing.
- Use the runtime-provided repository root (`/testbed` or a slot-scoped checkout) when executing commands.
