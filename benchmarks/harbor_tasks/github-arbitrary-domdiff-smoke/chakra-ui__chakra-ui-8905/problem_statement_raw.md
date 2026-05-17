# Gradient Props Not Working as Expected with Text Component And bgClip="Text" prop

## Issue Description
### Description

When applying the gradients with the gradientFrom, gradientTo, and bgGradient(direction) props, and then applying bgClip="Text" on a Text or Heading component, no gradient is rendered.

### Link to Reproduction

https://codesandbox.io/p/sandbox/peaceful-ritchie-4xpdf3?file=%2Fsrc%2Findex.tsx%3A21%2C9&workspaceId=04c0dc82-137b-495b-9f48-3685476ec068

### Steps to reproduce

1). Go to code sandbox link
2). Examine the lack of gradient on the text and heading components provided despite appropriate gradient props.

### Chakra UI Version

3.0.0-next.26

### Browser

Version 129.0.6668.60

### Operating System

- [ ] macOS
- [X] Windows
- [ ] Linux

### Additional Information

![gradientNotWorkingChakra](https://github.com/user-attachments/assets/d6400147-a0df-4804-b13a-a89cd55bc9f5)

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