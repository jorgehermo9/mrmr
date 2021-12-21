# mrmr
Maximum Relevance - Minimum redundancy feature selection algorithm implemented in Rust.

## Usage

CLI app.
It gets input from a csv dataset **with headers**.

> File is gotten from stdin if no csv file specified.

### Examples

```console
cd mrmr
cargo build

./target/debug/mrmr --help
./target/debug/mrmr -c dataset.csv -n 10
./target/debug/mrmr --csv dataset.csv --num-features 10 --class rank
```
