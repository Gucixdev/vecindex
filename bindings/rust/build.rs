//! Compiles vecindex's C (src/vecindex.c, one unity file) into a static library with the system's C compiler - no
//! build dependency. The sources are the repository's, two levels up; a packaged crate carries them in csrc/.
//! CC picks the compiler (default cc); for the machine it is built on (not when cross-compiling) -march=native.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = [here.join("../.."), here.join("csrc")]
        .iter()
        .find(|d| d.join("src/vecindex.c").is_file())
        .cloned()
        .expect("vecindex: no C sources (src/vecindex.c) beside this crate");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let obj = out.join("vecindex.o");
    let mut c = Command::new(&cc);
    c.args(["-O2", "-std=gnu99", "-fPIC", "-fvisibility=hidden", "-Wno-unused-function", "-c"])
        .arg(format!("-I{}", root.join("include").display()))
        .arg(format!("-I{}", root.join("src").display()))
        .arg(root.join("src/vecindex.c"))
        .arg("-o")
        .arg(&obj);
    if env::var("TARGET").ok() == env::var("HOST").ok() && env::var("VECINDEX_PORTABLE").ok().as_deref() != Some("1") {
        c.arg("-march=native");
    }
    run(&mut c, "compile src/vecindex.c");
    let lib = out.join("libvecindex.a");
    let _ = std::fs::remove_file(&lib);
    run(Command::new(&ar).arg("crs").arg(&lib).arg(&obj), "archive libvecindex.a");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vecindex");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    for d in &["src", "include"] {
        rerun(&root.join(d));
    }
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=VECINDEX_PORTABLE");
}

fn run(c: &mut Command, what: &str) {
    match c.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("vecindex: {} failed ({})", what, s),
        Err(e) => panic!("vecindex: {}: cannot run the C compiler or ar: {} (set CC / AR)", what, e),
    }
}

fn rerun(dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            println!("cargo:rerun-if-changed={}", e.path().display());
        }
    }
}
