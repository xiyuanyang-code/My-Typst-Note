use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "create" => {
                if args.len() > 2 {
                    let title = &args[2];
                    let filename = format!("{}.typ", title);

                    let template_path = Path::new("template/empty.typ");
                    match fs::read_to_string(template_path) {
                        Ok(content) => {
                            let new_content = content.replace(
                                "title: \"Basic Formatting Style for Typst\"",
                                &format!("title: \"{}\"", title)
                            );

                            match fs::write(&filename, new_content) {
                                Ok(_) => println!("Successfully created note: {}", filename),
                                Err(e) => eprintln!("Error writing file: {}", e),
                            }
                        }
                        Err(e) => eprintln!("Error reading template file: {}", e),
                    }
                } else {
                    println!("Usage: typstnote m \"<note title>\"");
                }
            }
            "compile" => {
                if args.len() > 2 {
                    let filename = &args[2];
                    let status = Command::new("typst")
                        .arg("compile")
                        .arg(filename)
                        .status();

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
                    println!("Usage: typstnote compile <filename.typ>");
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
    println!("  typstnote m \"<note title>\"      - Create a new note");
    println!("  typstnote compile <filename.typ> - Compile a note to PDF");
}