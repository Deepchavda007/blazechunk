//! Throughput benchmarks for the five high-level chunkers.
//!
//! Run with: `cargo bench --bench chunkers`

use chunk::{
    CharCounter, CodeChunker, Language, RecursiveChunker, RowCounter, SentenceChunker,
    TableChunker, TokenChunker,
};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Build roughly `target` bytes of varied ASCII prose.
fn make_prose(target: usize) -> String {
    let para = "The quick brown fox jumps over the lazy dog. \
                Chunking splits text at semantic boundaries such as periods, \
                question marks, and newlines? It does so extremely fast! Short. \
                Then a longer, winding sentence meanders through several clauses.\n\n";
    let reps = target / para.len() + 1;
    let mut s = para.repeat(reps);
    s.truncate(target); // `para` is ASCII, so every byte index is a char boundary
    s
}

/// A markdown table with `rows` data rows and a repeated header.
fn make_markdown_table(rows: usize) -> String {
    let mut s = String::from("| Name | Score | Notes |\n|------|-------|-------|\n");
    for i in 0..rows {
        s.push_str(&format!(
            "| user{i} | {} | note number {i} here |\n",
            i % 100
        ));
    }
    s
}

/// An HTML table equivalent to [`make_markdown_table`].
fn make_html_table(rows: usize) -> String {
    let mut s = String::from("<table><thead><tr><th>Name</th><th>Score</th></tr></thead><tbody>");
    for i in 0..rows {
        s.push_str(&format!("<tr><td>user{i}</td><td>{}</td></tr>", i % 100));
    }
    s.push_str("</tbody></table>");
    s
}

/// A real, multi-function source file from this repo (falls back to synthetic
/// generated code so the benchmark still runs from any working directory).
fn code_input() -> String {
    std::fs::read_to_string("src/split.rs").unwrap_or_else(|_| {
        (0..400)
            .map(|i| {
                format!(
                    "fn func_{i}(a: i32, b: i32) -> i32 {{\n    let x = a + b;\n    x * {i}\n}}\n\n"
                )
            })
            .collect()
    })
}

fn bench_prose_chunkers(c: &mut Criterion) {
    let small = make_prose(50 * 1024); // ~50 KB
    let large = make_prose(1024 * 1024); // ~1 MB

    for (label, text) in [("50KB", &small), ("1MB", &large)] {
        let mut group = c.benchmark_group(format!("prose_{label}"));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("RecursiveChunker/2048", |b| {
            b.iter(|| {
                black_box(
                    RecursiveChunker::new()
                        .chunk_size(2048)
                        .chunk(black_box(text.as_str()), &CharCounter)
                        .unwrap(),
                )
            })
        });

        group.bench_function("SentenceChunker/2048", |b| {
            b.iter(|| {
                black_box(
                    SentenceChunker::new()
                        .chunk_size(2048)
                        .chunk(black_box(text.as_str()), &CharCounter)
                        .unwrap(),
                )
            })
        });

        group.bench_function("TokenChunker/2048", |b| {
            b.iter(|| {
                black_box(
                    TokenChunker::new()
                        .chunk_size(2048)
                        .chunk(black_box(text.as_str()), &CharCounter)
                        .unwrap(),
                )
            })
        });

        group.finish();
    }
}

fn bench_table_chunker(c: &mut Criterion) {
    let md = make_markdown_table(2000);
    let html = make_html_table(2000);

    let mut group = c.benchmark_group("table_markdown");
    group.throughput(Throughput::Bytes(md.len() as u64));
    group.bench_function("rows50", |b| {
        b.iter(|| {
            black_box(
                TableChunker::new()
                    .chunk_size(50)
                    .chunk(black_box(md.as_str()), &RowCounter)
                    .unwrap(),
            )
        })
    });
    group.finish();

    let mut group = c.benchmark_group("table_html");
    group.throughput(Throughput::Bytes(html.len() as u64));
    group.bench_function("rows50", |b| {
        b.iter(|| {
            black_box(
                TableChunker::new()
                    .chunk_size(50)
                    .chunk(black_box(html.as_str()), &RowCounter)
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn bench_code_chunker(c: &mut Criterion) {
    let code = code_input();
    let mut group = c.benchmark_group("code_rust");
    group.throughput(Throughput::Bytes(code.len() as u64));
    for size in [512usize, 2048] {
        group.bench_function(format!("chunk_size_{size}"), |b| {
            b.iter(|| {
                black_box(
                    CodeChunker::new()
                        .chunk_size(size)
                        .language(Language::Rust)
                        .chunk(black_box(code.as_str()), &CharCounter),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_prose_chunkers,
    bench_table_chunker,
    bench_code_chunker
);
criterion_main!(benches);
