# My Typst Note

## Introduction

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## Template

We use [dvdtyp](https://github.com/DVDTSB/dvdtyp) as the template for notes. Really awesome template!

> Several modifications based on personal needs are made.

## Installation

First, ensure you have [Rust](https://www.rust-lang.org/tools/install) and [Cargo](https://doc.rust-lang.org/cargo/) installed.

For Linux system or WSL, just run the quick scripts below to install all the dependencies.

```bash
bash scripts/install.sh
```

<details>
<summary>
If you want it to be installed manually...
</summary>

Firstly, install typst-cli using cargo:

```bash
cargo install typst-cli
```

From the project root, build the self-contained executable:

```bash
cargo build --release
```

Finally, install the executable by moving it to a directory in your system's PATH. This makes the `note` command available system-wide. For example:

```bash
sudo cp target/release/note /usr/local/bin/
# target/release/note is the built binary file.
# you can use it directly in your command line 
```
</details>

Then restart your terminal with `source ~/.zshrc`, etc.

## Usage

To create a new note, run the following command, replacing `"<note title>"` with your desired title:

```bash
note new "<note title>"
```

This will create a new `<note title>.typ` file in the current directory.

To compile a note into a PDF, run:

```bash
note pdf "<note title>"
```

For example:

```bash
note new algebra
# it will create a new algebra.typ file in current working directory

note pdf algebra
# compile algebra.typ into algebra.pdf
```
