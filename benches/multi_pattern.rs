use chunk::chunk;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn bench_patterns_api(c: &mut Criterion) {
    let text = std::fs::read("benches/data/enwik8").expect("Failed to load enwik8.");

    let mut group = c.benchmark_group("patterns_api_enwik8");
    group.throughput(Throughput::Bytes(text.len() as u64));

    // Baseline: delimiters only (existing API)
    group.bench_function("delimiters_only", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(4096)
                .delimiters(b"\n.?!")
                .collect();
            black_box(chunks)
        })
    });

    // New API: delimiters + 3 CJK patterns (memmem path)
    group.bench_function("delimiters_plus_3_patterns", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(4096)
                .delimiters(b"\n.?!")
                .patterns(&["。", "，", "！"])
                .collect();
            black_box(chunks)
        })
    });

    // New API: delimiters + 5 CJK patterns (Aho-Corasick path)
    group.bench_function("delimiters_plus_5_patterns", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(4096)
                .delimiters(b"\n.?!")
                .patterns(&["。", "，", "！", "？", "；"])
                .collect();
            black_box(chunks)
        })
    });

    // New API: delimiters + 3 patterns + forward_fallback
    group.bench_function("delimiters_3pat_fwd_fallback", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(4096)
                .delimiters(b"\n.?!")
                .patterns(&["。", "，", "！"])
                .forward_fallback()
                .collect();
            black_box(chunks)
        })
    });

    // Patterns only (no delimiters)
    group.bench_function("patterns_only_3", |b| {
        b.iter(|| {
            let chunks: Vec<_> = chunk(black_box(&text))
                .size(4096)
                .delimiters(b"")
                .patterns(&["。", "，", "！"])
                .collect();
            black_box(chunks)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_patterns_api);
criterion_main!(benches);
