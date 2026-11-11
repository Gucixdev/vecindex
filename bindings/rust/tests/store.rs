//! The crate against a real store on disk: cargo test (from bindings/rust).
use std::path::PathBuf;
use vecindex::{key, Db, Filter};

const DIM: usize = 8;

fn vec(i: i64) -> Vec<f32> {
    (0..DIM).map(|d| ((i as f32) * 1.3 + d as f32 * 0.7).sin() + if d as i64 == i % DIM as i64 { 0.5 } else { 0.0 }).collect()
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("vecindex_rs_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn add_search_reopen() {
    let d = tmp("reopen");
    let p = d.join("t.vxi");
    {
        let db = Db::open(&p, DIM).unwrap();
        for i in 1..=200 {
            if i % 5 == 0 {
                db.add_text(i, &vec(i), format!("chunk {}", i)).unwrap();
            } else {
                db.add(i, &vec(i)).unwrap();
            }
        }
        assert_eq!(db.len(), 200);
        let hits = db.search(&vec(42), 3).unwrap();
        assert_eq!(hits[0].0, 42);
        assert!(hits[0].1 > 0.99);
        assert_eq!(db.text(15).as_deref(), Some("chunk 15"));
        assert_eq!(db.text(16), None);
        db.sync().unwrap();
    }
    let db = Db::open(&p, 0).unwrap();
    assert_eq!(db.dim(), DIM);
    assert_eq!(db.len(), 200);
    assert_eq!(db.search(&vec(7), 1).unwrap()[0].0, 7);
    assert_eq!(db.text(20).as_deref(), Some("chunk 20"));
    assert!(db.get(9).is_some() && db.get(999).is_none());
    drop(db);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn errors() {
    let db = Db::memory(DIM).unwrap();
    assert!(db.add(1, &[1.0, 2.0]).is_err());
    let e = db.add(1, &[f32::NAN; DIM]).unwrap_err();
    assert!(e.0.contains("NaN") || e.0.contains("finite"), "{}", e);
    assert!(!db.delete(99));
    assert!(db.search_where(None, 5, None).is_err());
    let d = tmp("bad");
    let p = d.join("bad.vxi");
    std::fs::write(&p, b"not a store at all, but some bytes").unwrap();
    assert!(Db::open(&p, 0).is_err());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn conditions() {
    let db = Db::memory(DIM).unwrap();
    let (even, odd) = (key("even"), key("odd"));
    for i in 1..=300 {
        db.add_text(i, &vec(i), if i == 77 { "the rare word zyxwv".to_string() } else { format!("plain text {}", i) }).unwrap();
        db.set_num(i, i * 10).unwrap();
        db.set_ns(i, if i % 2 == 0 { even } else { odd }).unwrap();
        if i % 3 == 0 {
            db.add_labels(i, if i % 6 == 0 { "dog" } else { "cat" }).unwrap();
        }
    }
    db.label("dog", Some("pet")).unwrap();
    db.label("cat", Some("pet")).unwrap();
    let q = vec(150);
    assert!(db.search_where(Some(&q), 20, Some(Filter::Ns(even))).unwrap().iter().all(|h| h.0 % 2 == 0));
    assert_eq!(db.count_ns(odd), 150);
    assert_eq!(db.search_where(Some(&q), 20, Some(Filter::AnyNs(&[odd, even]))).unwrap()[0].0, 150);
    assert!(db.search_where(Some(&q), 20, Some(Filter::Num(1000, 1500))).unwrap().iter().all(|h| (100..=150).contains(&h.0)));
    let pets = db.search_where(Some(&q), 50, Some(Filter::Labels { all: "pet", any: "", none: "" })).unwrap();
    assert!(!pets.is_empty() && pets.iter().all(|h| h.0 % 3 == 0));
    assert_eq!(db.search_where(None, 1, Some(Filter::Text("zyxwv"))).unwrap()[0].0, 77);
    db.update(150, &vec(151)).unwrap();
    assert_eq!(db.num(150), Some(1500));
    assert_eq!(db.delete_ns(odd), 150);
    assert_eq!(db.len(), 150);
}

#[test]
fn grouped_and_threads() {
    let db = std::sync::Arc::new(Db::memory(DIM).unwrap());
    for doc in 1..=5i64 {
        for chunk in 0..4 {
            db.add(doc << 20 | chunk, &vec(doc * 10 + chunk)).unwrap();
        }
    }
    let hits = db.search_where(Some(&vec(30)), 3, Some(Filter::Group(20))).unwrap();
    let docs: Vec<i64> = hits.iter().map(|h| h.0 >> 20).collect();
    assert_eq!(docs[0], 3);
    let mut seen = docs.clone();
    seen.dedup();
    assert_eq!(seen.len(), docs.len());
    let ts: Vec<_> = (0..4)
        .map(|t| {
            let db = db.clone();
            std::thread::spawn(move || {
                for i in 0..50 {
                    db.add(1000 + t * 100 + i, &vec(1000 + t * 100 + i)).unwrap();
                    assert!(!db.search(&vec(i), 3).unwrap().is_empty());
                }
            })
        })
        .collect();
    for t in ts {
        t.join().unwrap();
    }
    assert_eq!(db.len(), 20 + 200);
}

#[test]
fn version_and_key() {
    assert_eq!(vecindex::version(), env!("CARGO_PKG_VERSION"));
    assert_eq!(key("acme"), key("acme"));
    assert_ne!(key("acme"), key("acmf"));
}
