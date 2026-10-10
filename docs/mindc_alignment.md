# Toolchain notes: mindc 0.10.x

This repository targets **mindc 0.10.x**. CI pins the public `v0.10.2`
release archives from <https://github.com/star-ga/mind/releases> by sha256.

## What changed from the earlier 0.2.x-era sources

The previous `src/` files used a draft surface (`module { ... }` blocks,
draft `[...]` module attributes, `Vec`, slices, `q16_16`
methods, `unimplemented!()`) that current mindc does not accept:
`mindc build` refused them (`unknown identifier`, `expected for in impl`)
and no binary could be produced. The current `src/bitnet.mind` is written in
the executable subset that mindc 0.10.x compiles end to end:

| Need | Used here |
|---|---|
| Fixed-point values | `i64` holding Q16.16; multiply then `>> 16` |
| Arrays / matrices | heap buffers via `__mind_alloc`, `__mind_load_i64`, `__mind_store_i64`, `__mind_load_i8`, `__mind_store_i8` |
| Ternary weights | 2-bit codes packed 4 per byte, decoded with shifts and masks |
| Hashing | `import std.sha256;` (`sha256(in, len, out)`) |
| Output | `import std.io;` (`print_bytes(buf, len)`) |
| Governance | `#[deterministic]` on every function except `main` |

String literals are outside the executable subset, so printed text is
written as byte codes.

## Build modes

| Command | Needs | Notes |
|---|---|---|
| `mindc build --release --emit cdylib --out libbitnet.so` | `cc`, `ld`, `as`, `objcopy` | used by the harness and CI |
| `mindc build --release` | MIND CPU runtime library | produces an executable |
| `mindc src/bitnet.mind --emit-evidence a.mic3` | nothing extra | mic@3 artifact for `mindc verify` |

The pure-MIND native ELF backend (`mindc build --backend native`) does not
yet lower the heap intrinsics this program uses, so the harness uses the
default backend.

## Known limits of the `#[deterministic]` check (observed with 0.10.2)

The check rejects direct calls to the compiler's known nondeterministic
builtins (for example an unseeded `rand_int()`), which is what
`examples/governance/rng_in_kernel.mind` exercises. In the published
0.10.2 release it did not
flag a `#[deterministic]` function that reaches the wall clock through an
imported std function (`import std.time;` then `now_ns()`), or through an
unannotated helper function in the same file. A newer development build of
mindc (same 0.10.2 version string, not yet a published release) refuses
both: the first through the `#[deterministic]` check, the second through
the default determinism gate on artifact emission. Until that ships, keep
every function on the forward path annotated, as `src/bitnet.mind` does, and
do not rely on the check to follow calls it does not see.

## What is and is not pinned

The program output (`TOKEN` / `ROOT` lines) is pinned in
`results/expected.json` and must match on every compiler build and host. The
mic@3 `trace_hash` is not pinned: it hashes the compiler's canonical IR, and
two builds of mindc that both report 0.10.2 produced different IR for this
source (published release: `c457cf7e...`; newer development build:
`d6c4124f...`). The harness therefore checks that each compiler reproduces
its own artifact byte for byte and that `mindc verify` accepts it.
