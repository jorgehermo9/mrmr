# mrmr
Maximum Relevance - Minimum redundancy feature selection algorithm implemented in Rust.

## Usage

CLI app.
It gets input from a csv dataset **with headers**.Dataset should be previously discretized.

> File is gotten from stdin if no csv file specified.

## Build & Examples

### Build

```console
cd mrmr
#release flag for optimization
cargo build --release
```

### Examples

```console
./target/debug/mrmr --help
./target/debug/mrmr -c dataset.csv -n 10
./target/debug/mrmr --csv dataset.csv --num-features 10 --class rank
```
