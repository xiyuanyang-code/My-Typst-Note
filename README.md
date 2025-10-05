# My Typst Note

## Introduction

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## File Structure

```text
.
├── Cargo.lock
├── Cargo.toml
├── LectureNote
│   ├── AlgorithmsNote
│   │   ├── AI1804-Algorithm.typ
│   │   ├── Leetcode.typ
│   │   ├── MIT6006-DP.typ
│   │   ├── MIT6046J-Algorithm.typ
│   │   └── tex_note
│   │       └── code_demo.tex
│   ├── MachineLearning
│   │   ├── AI1811-ML.typ
│   │   └── ML
│   ├── NumericalAnalysis
│   │   ├── AI1807-Numerical.typ
│   │   └── src
│   │       └── demo.py
│   ├── Probability
│   │   └── MATH1207-Probability.typ
│   └── RL
│       └── CS234-RL.typ
├── Makefile
├── run.sh
├── scripts
│   ├── install.sh
│   ├── refresh.py
│   └── update_notes
│       └── compile.py
├── src
│   └── main.rs
└── template
    ├── backup.typ
    ├── empty.typ
    └── format.typ
```

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

Finally, install the executable locally using `cargo install --path .`:

```bash
# install locally
cargo install --path .
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

## Lecture Notes

All my lecture notes written in typst will be stored in `./LectureNote` folder, including there courses...

For all the lecture notes, see [Release Page](https://github.com/xiyuanyang-code/My-Typst-Note/releases/latest) for more information!

Finished Courses:

- MIT 6.006 Dynamic Programming

Current Updating Courses:

- SJTU-AI1807 Numerical Analysis
- SJTU-AI1804 Analysis of Algorithms
- LeetCode Daily Problems
- SJTU-AI1811 Machine Learning
- SJTU-MATH1207 Probability
- MIT6046J Analysis of Algorithms
- CS234 Reinforcement Learning