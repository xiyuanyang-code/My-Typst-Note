use std::env;
use std::fs;
use std::process::Command;
const TEMPLATE_CONTENT: &str = include_str!("../template/empty.typ");
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "new" => {
                if args.len() > 2 {
                    let title = &args[2];
                    let filename = format!("{}.typ", title);
                    let new_content = TEMPLATE_CONTENT.replace(
                        "title: \"Basic Formatting Style for Typst\"",
                        &format!("title: \"{}\"", title),
                    );
                    match fs::write(&filename, new_content) {
                        Ok(_) => println!("Successfully created note: {}", filename),
                        Err(e) => eprintln!("Error writing file: {}", e),
                    }
                } else {
                    println!("Usage: note new \"<note title>\"");
                }
            }
            "pdf" => {
                if args.len() > 2 {
                    let title = &args[2];
                    let filename = format!("{}.typ", title);
                    let status = Command::new("typst").arg("compile").arg(&filename).status();
                    match status {
                        Ok(status) => {
                            if status.success() {
                                println!("Successfully compiled {}", filename);
                            } else {
                                eprintln!("Error compiling {}", filename);
                            }
                        }
                        Err(e) => eprintln!("Error executing typst command: {}", e),
                    }
                } else {
                    println!("Usage: note pdf <filename.typ>");
                }
            }
            _ => print_usage(),
        }
    } else {
        print_usage();
    }
}
fn print_usage() {
    println!("Usage:");
    println!("  note new \"<note title>\"      - Create a new note");
    println!("  note pdf <filename>            - Compile a note to PDF");
}
