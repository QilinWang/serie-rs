# SERIE-RS 🦀

**Controlled time-series data, built in Rust.** Generate trajectories,
introduce parameter or state shocks, and reuse noise draws for paired comparisons.

- [Project home, real Lorenz shock and examples](https://github.com/QilinWang/serie-rs)
- [Features, validation and provenance](https://github.com/QilinWang/serie-rs/blob/main/docs/recovery-and-validation.md)
- [Python companion and interactive demo](https://github.com/QilinWang/serie)

In a repository checkout, the CLI is enabled explicitly:

```sh
cargo run -p serie-rs --features cli -- generate lorenz --shock param --sigma 0.1
```

This recovered Rust snapshot is separate from the current Fern Python data
path. Qilin Wang directed the research and system work; AI coding agents
contributed to the broader implementation and recovery. See the project home
for tested scope and details.

MIT — see [LICENSE](LICENSE).
