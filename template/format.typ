#import "@preview/dvdtyp:1.0.1": *
#show: dvdtyp.with(
  title: "Basic Formatting Style for Typst",
  author: "Xiyuan Yang",
  abstract: [This is a simple formatting style],
)
#show link: set text(fill: blue, weight: 700)
#show link: underline
#show ref: set text(fill: blue, weight: 500)


// comment for hiding outlines
#outline()

// uncomment for changing to a new page
// #pagebreak()

= Introduction

For detailed reference: See https://typst-doc-cn.github.io/tutorial/ for more info.

#link("https://typst-doc-cn.github.io/tutorial/", "Official Tutorial").

== Basic Syntax

This is a file, we want to make it very pretty.

- We want to make this text *bold*.

- We want to make this text _italic_.

- We want to make this text #raw("a raw content like the code") or just simply using `raw code`.

#remark("Better fit for writing math notes")[
  Thus, this tool is useful to write math notes than write codes.
]


== Advanced Settings

=== Code Blocks

- `C++` Files

```cpp
#include <iostream>
using namespace std;
int main(){
  std::cout << "hello world" << std::endl;
  // just a demo program
}
```

- `Python` Files

```py
import torch
if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
```

- Rust Files:

```rs
fn main(){
	println!("Hello World!");
}
```

=== Images

Just the same as images in markdown and LaTeX.

#figure(
  image("/template/image.png", width: 70%),
  caption: [
    _Just a template_ from the screen shot.
  ],
) <img-test>

Reference to @img-test.

=== Tables <tables>

#align(center)[
  #table(
    columns: 2,
    rows: 2,
    align: center,
    [第一列，第一行], [第二列，第一行],
    [第一列，第二行], [第二列，第二行],
  )
]


#align(center)[
  #table(
    columns: 2,
    rows: 1,
    align: center,
    [dd], [],
  )
]

#figure(
  table(
    columns: 2,
    rows: 2,
    align: center,
    [第一列，第一行], [第二列，第一行],
    [第一列，第二行], [第二列，第二行],
  ),
) <tab-test>

Reference to @tab-test.


=== Several Windows

Several windows efficient to note taking.

Refer to @tables.

#align(center)[
  #table(
    columns: 2,
    align: center,
    [Problems],
    [#problem[
      Prove that $1+1=3$.
    ]],

    [Theorem],
    [#theorem("test theorem")[
      $ forall x: x^2 >= 0 $
    ]],

    [Lemma],
    [#lemma("test lemma")[
      test lemma
    ]],

    [Corollary],
    [#corollary("test corrllary")[
      test corollary
    ]],

    [Definition],
    [#definition("test definition")[
      i define hi as a greeting
    ]],

    [Proposition],
    [#proposition("test proposition")[
      test proposition
    ]],

    [Remark],
    [#remark("test remark")[
      test remark
    ]],

    [Observation],
    [#observation("test observation")[
      test observation
    ]],

    [Example],
    [#example("test example")[
      test example
    ]],
  )
]

=== Proof Blocks

#proof[
  $ "hi"="hello"="greeting" $
]

= Making own theorem enviorments

to make your own theorem enviorments, you can use the `builder-thmbox` and `builder-thmline` functions to generate _theorem styles_ and then use those to make theorems (idk if this is too convoluted or not, make an issue on github if you have a better idea).

```typ
#let theorem-style = builder-thmbox(color: colors.at(6), shadow: (offset: (x: 3pt, y: 3pt), color: luma(70%)))
#let theorem = theorem-style("theorem", "Theorem")
#let lemma = theorem-style("lemma", "Lemma")

#let definition-style = builder-thmline(color: colors.at(8))
#let definition = definition-style("definition", "Definition")
#let proposition = definition-style("proposition", "Proposition")

```

There is also a color pallete.

#{
  let nums = range(16)

  align(
    center,

    table(
      columns: 16,
      stroke: 0pt,
      inset: 0em,
      ..nums.map(i => rect([#i], fill: colors.at(i), width: 2em, height: 2em)),
    ),
  )
}
