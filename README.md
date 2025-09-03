# My Typst Note

## Introduction

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## Installations

Need to install Cargo first.

### Build From Source

First, build the release version of the application:

```bash
cargo build --release
```

Then, install the executable by moving it to a directory in your system's PATH. For example, you can move it to `/usr/local/bin`:

```bash
sudo cp target/release/note /usr/local/bin/
```

## Template

For the template, I use [dvdtyp](https://github.com/DVDTSB/dvdtyp).

- [Backup Configuration Files](./template/backup.typ)

- [Formatting Tutorial](./template/format.typ)

## Usage

To create a new note, run the following command, replacing `"<note title>"` with your desired title:

```bash
note --create "<note title>"
```

This will create a new `<note title>.typ` file in the current directory. The file's title will also be set to `<note title>`.

To compile a note into a PDF, run:

```bash
note --release "<note title>.typ"
```

## Todo List

- Finish Rust File for simple CLI tools usage.

- Create empty file for templating.