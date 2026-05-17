<!-- harbor_instruction_version: 1 -->
# chakra-ui__chakra-ui-8905

**Repository:** chakra-ui/chakra-ui

## Task Summary

Gradient Props Not Working as Expected with Text Component And bgClip="Text" prop

## Expected Behavior

# Gradient Props Not Working as Expected with Text Component And bgClip="Text" prop

## Issue Description
### Description

When applying the gradients with the gradientFrom, gradientTo, and bgGradient(direction) props, and then applying bgClip="Text" on a Text or Heading component, no gradient is rendered.

### Link to Reproduction

### Steps to reproduce

1). Go to code sandbox link
2). Examine the lack of gradient on the text and heading components provided despite appropriate gradient props.

### Chakra UI Version

3.0.0-next.26

### Browser

Version 129.0.6668.60

### Operating System

- [X] Windows

### Additional Information

## Fix: fix: backgroundClip text transform

## Additional Context
Closes #8904

## 📝 Description

Added transform function to `backgroundClip` property to appropriately handle `text` value by setting `color` to `transparent`

## ⛳️ Current behavior (updates)

`bgGradient` property did not work on typography components since color was not set to transparent when `bgClip` was set to `text`

## 🚀 New behavior

Transform function added to `backgroundClip` property to appropriately set color to transparent when backgroundClip is set to `text`

## 💣 Is this a breaking change (Yes/No):

No. All tests still pass. This just resolves a bug related to setting gradient on typography components and preserves backwards compatibility.

## 📝 Additional Information

New `Heading` story created showcasing gradient on heading.

## Likely Fix Direction

Implement the intended UI behavior described by the task.

## Likely Touch Points

- `.changeset/short-lions-jog.md`
- `apps/compositions/src/examples/heading-with-gradient.tsx`
- `packages/react/__stories__/heading.stories.tsx`
- `packages/react/src/preset-base.ts`

## Success Criteria

- Fix the bug described above without breaking the existing preview.
- Ensure the rendered preview at `/iframe.html?id=typography-heading--basic&viewMode=story` shows the intended behavior.
- Keep the repository build and tests passing.
- Use the runtime-provided repository root (`/testbed` or a slot-scoped checkout) when executing commands.
