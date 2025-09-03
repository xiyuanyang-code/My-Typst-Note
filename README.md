# My Typst Note

## Introduction

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## Installations

Need to install Cargo and Typst first.

## Template

For the template, I use [dvdtyp](https://github.com/DVDTSB/dvdtyp).

- [Backup Configuration Files](./template/backup.typ)

- [Formatting Tutorial](./template/format.typ)

## Usage

To create a new note, run the following command, replacing `"<note title>"` with your desired title:

```bash
cargo run -- m "<note title>"
```

This will create a new `<note title>.typ` file in the current directory. The file's title will also be set to `<note title>`.

To compile a note into a PDF, run:

```bash
cargo run -- compile "<note title>.typ"
```

## Todo List

- Finish Rust File for simple CLI tools usage.

- Create empty file for templating.