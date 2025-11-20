# My Typst Note

A simple CLI tool for quickly generating formatted notes written in typst and deploying them in PDF.

## File Structure
<!-- INSERT BEGIN -->
```text
.
├── Cargo.lock
├── Cargo.toml
├── LectureNote
│   ├── AlgorithmsNote
│   │   ├── AI1804-Algorithm.typ
│   │   ├── lecture_pdf
│   │   │   ├── notes
│   │   │   └── slides
│   │   ├── Leetcode.typ
│   │   ├── MIT6006-DP.typ
│   │   ├── MIT6046J-Algorithm.typ
│   │   ├── src
│   │   │   ├── difference_const
│   │   │   │   ├── difference_const_simple.py
│   │   │   │   ├── difference_const.py
│   │   │   │   └── test_simple.py
│   │   │   ├── homework_1
│   │   │   │   ├── homework1_assignment
│   │   │   │   │   ├── problem
│   │   │   │   │   │   ├── 题目描述.md
│   │   │   │   │   │   ├── 1_1_Maximum_Subarray.py
│   │   │   │   │   │   ├── 1_2_Find_Peak_Element.py
│   │   │   │   │   │   ├── 1_3_Majority_Element.py
│   │   │   │   │   │   ├── 1_4_Find_the_Duplicate_Number.py
│   │   │   │   │   │   ├── 2_1_Kth_Largest_Element.py
│   │   │   │   │   │   ├── 3_1_Hospital_Clinic_Allocation_System.py
│   │   │   │   │   │   └── Problem Description.md
│   │   │   │   │   ├── README.md
│   │   │   │   │   └── test
│   │   │   │   │       ├── test_1_1_maximum_subarray.py
│   │   │   │   │       ├── test_1_2_find_peak_element.py
│   │   │   │   │       ├── test_1_3_majority_element.py
│   │   │   │   │       ├── test_1_4_find_the_duplicate_number.py
│   │   │   │   │       ├── test_2_1_kth_largest_element.py
│   │   │   │   │       ├── test_3_1_hospital_clinic_allocation_system.py
│   │   │   │   │       └── test_utils.py
│   │   │   │   └── homework1_original
│   │   │   │       ├── problem_2_1_1_two_sum.py
│   │   │   │       ├── problem_2_1_2_max_subarray.py
│   │   │   │       ├── problem_2_1_3_peak_element.py
│   │   │   │       ├── problem_2_1_4_majority_element.py
│   │   │   │       ├── problem_2_2_sorted_set.py
│   │   │   │       ├── problem_2_3_merge_sort_inversions.py
│   │   │   │       ├── problem_2_4_hospital_system.py
│   │   │   │       └── README.md
│   │   │   ├── quiz_2
│   │   │   │   ├── problem_3_1.py
│   │   │   │   └── problem_3_2.py
│   │   │   └── quiz_4
│   │   │       ├── 1.py
│   │   │       ├── 2.py
│   │   │       ├── codebase.py
│   │   │       └── README.md
│   │   └── tex_note
│   │       └── code_demo.tex
│   ├── DeepLearning
│   │   ├── AI1811-DL.typ
│   │   ├── homework
│   │   │   ├── hm1.md
│   │   │   └── hm1.py
│   │   └── slides
│   ├── MachineLearning
│   │   ├── AI1811-ML.typ
│   │   ├── hm
│   │   │   ├── coding.py
│   │   │   ├── decision-tree.py
│   │   │   ├── pca-2.py
│   │   │   ├── pca.py
│   │   │   └── Source Code for Problem7 PCA.md
│   │   ├── src
│   │   │   └── clustering.py
│   │   └── src_scripts
│   │       ├── classification.md
│   │       ├── clustering_full.md
│   │       ├── clustering.md
│   │       ├── Dimensionality_reduction.md
│   │       ├── intro.md
│   │       ├── Model_selection.md
│   │       ├── Parameter_Estimation.md
│   │       ├── regression_full.md
│   │       └── regression.md
│   ├── NumericalAnalysis
│   │   ├── AI1807-Numerical.typ
│   │   ├── lecture-slides
│   │   └── src
│   │       ├── demo.py
│   │       ├── fp.py
│   │       ├── homework
│   │       │   ├── hm1.py
│   │       │   ├── hm2.py
│   │       │   └── visualize_resample.py
│   │       ├── interpolation_for_torch.py
│   │       ├── interpolation.py
│   │       └── spline.py
│   ├── Probability
│   │   └── MATH1207-Probability.typ
│   ├── RL
│   │   ├── AI1811-RL.typ
│   │   └── CS234-RL.typ
│   └── SoftwareDevelopment
│       ├── README.md
│       └── src
│           ├── behave_test
│           │   └── features
│           │       ├── demo.feature
│           │       ├── environment.py
│           │       ├── login.feature
│           │       ├── register.feature
│           │       └── steps
│           │           ├── common_steps.py
│           │           ├── demo.py
│           │           ├── login_steps.py
│           │           └── register_steps.py
│           └── test_demo
│               ├── pom.xml
│               └── src
│                   ├── main
│                   │   └── java
│                   │       └── Example.java
│                   └── test
│                       └── java
│                           └── MainTest.java
├── Makefile
├── README.md
├── result
│   ├── AI1804-Algorithm.pdf
│   ├── AI1807-Numerical.pdf
│   ├── AI1811-DL.pdf
│   ├── AI1811-ML.pdf
│   ├── AI1811-RL.pdf
│   ├── alphabuild.pdf
│   ├── CS234-RL.pdf
│   ├── Leetcode.pdf
│   ├── MATH1207-Probability.pdf
│   ├── MIT6006-DP.pdf
│   └── MIT6046J-Algorithm.pdf
├── run.sh
├── scripts
│   ├── install.sh
│   ├── refresh.py
│   ├── update_files.py
│   └── update_notes
│       └── compile.py
├── src
│   └── main.rs
└── template
    ├── backup.typ
    ├── empty.typ
    └── format.typ

45 directories, 105 files

```
<!-- INSERT LAST -->

## Template

We use [dvdtyp](https://github.com/DVDTSB/dvdtyp) as the template for notes. Really awesome template!

> Several modifications based on personal needs are made.

## Usage for Typst CLI tools

### Installation

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

### Usage

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

All my lecture notes written in typst will be stored in `./LectureNote` folder. All open source. You can see [Github Release Page](https://github.com/xiyuanyang-code/My-Typst-Note/releases/latest) for more information!

Finished Courses:

- MIT 6.006 Dynamic Programming

Current Updating Courses:

- SJTU-AI1807 Numerical Analysis
- SJTU-AI1804 Analysis of Algorithms
- LeetCode Daily Problems
- SJTU-AI1811 Machine Learning
- SJTU-AI1811 Deep Learning
- SJTU-MATH1207 Probability
- MIT6046J Analysis of Algorithms
- CS234 Reinforcement Learning