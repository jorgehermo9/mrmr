# mrmr
Simple Maximum Relevance - Minimum redundancy feature selection algorithm implemented in Rust.
Federated learning version can be found at [fed-mRMR](https://github.com/jorgehermo9/fed-mrmr)

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
> Executable is generated to ./target/release/mrmr

### Execution with cargo

```console
cargo run --release -- -c daset.csv -n 10
```
### Examples

```console
./target/release/mrmr --help
./target/release/mrmr -c dataset.csv -n 10
./target/release/mrmr --csv dataset.csv --num-features 10 --class rank
```
