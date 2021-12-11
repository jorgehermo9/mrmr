# mrmr
Maximum Relevance - Minimum redundancy feature selection algorithm implemented in Rust.

## Usage

There is currently no cli args interface, only basic usage.
It gets input from a csv datasets **with headers**.The class
must be named `class` in the headers. 

> File is gotten from stdin.

###example

```console
cargo run < dataset.csv
```
