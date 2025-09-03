# My Typst Note

## Introduction

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## Installation

First, ensure you have [Rust](https://www.rust-lang.org/tools/install) and [Typst](https://github.com/typst/typst#installation) installed.

Then, from the project root, build the self-contained executable:

```bash
cargo build --release
```

Finally, install the executable by moving it to a directory in your system's PATH. This makes the `note` command available system-wide. For example:

```bash
sudo cp target/release/note /usr/local/bin/
```

## Usage

To create a new note, run the following command, replacing `"<note title>"` with your desired title:

```bash
note new "<note title>"
```

This will create a new `<note title>.typ` file in the current directory.

To compile a note into a PDF, run:

```bash
note pdf "<note title>.typ"
```

## Todo List

- Finish Rust File for simple CLI tools usage.

- Create empty file for templating.
