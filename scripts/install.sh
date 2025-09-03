#!/bin/bash
# ! Make sure you have installed cargo first!

echo "Installing typst-cli first!"
cargo install typst-cli

echo "Now installing and building projects..."
cargo build --release

sudo cp target/release/note /usr/local/bin/
