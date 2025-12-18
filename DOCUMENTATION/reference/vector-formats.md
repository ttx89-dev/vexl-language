# VEXL Universal Vector Formats Specification

## Overview

VEXL implements a universal vector serialization system where **everything is a vector**. This enables "vector-everywhere" persistence, allowing any data structure to be serialized and deserialized as vectors. The system provides multiple format families with different trade-offs for various use cases.

## Format Families (.x* Extensions)

### Binary Format (.xd) - Vector Data
**Purpose:** High-performance binary serialization for production use
**Use Cases:** Storage, IPC, network protocols, performance-critical applications

### Text Format (.xt) - Vector Text
**Purpose:** Human-readable serialization for debugging and configuration
**Use Cases:** Configuration files, debugging output, interchange with other tools

### Graphics Format (.xg) - Vector Graphics
**Purpose:** Visual representation of code/data as vector graphics
**Use Cases:** Visual programming, documentation, educational tools

### Archive Format (.xa) - Vector Archive
**Purpose:** Compressed multi-vector containers
**Use Cases:** Package management, data bundles, offline storage

### Stream Format (.xs) - Vector Stream
**Purpose:** Streaming vector protocols for real-time data
**Use Cases:** Network protocols, live data feeds, streaming analytics

---

## Binary Format (.xd) Specification

### Header Structure (64 bytes)

```
Offset | Size | Field | Description
-------|------|-------|-------------
0      | 4    | Magic | "VEXL" (0x5645584C)
4      | 2    | Version | Format version (current: 1)
6      | 2    | Reserved | Reserved for future use
8      | 8    | TypeTag | Element type identifier
16     | 8    | Dimensionality | Number of dimensions
24     | 64   | Shape | Dimension sizes (max 8 dimensions)
88     | 8    | StorageMode | Storage mode enum
96     | 8    | Metadata | Flags and compression info
104    | 8    | DataSize | Size of data section in bytes
112    | 8    | IndexSize | Size of index section (sparse only)
120    | 16   | Checksum | CRC32 of header
136    | 8    | Reserved | Reserved for future use
```

### Type Tags

| Value | Type | Description |
|-------|------|-------------|
| 0     | i64  | 64-bit signed integer |
| 1     | f64  | 64-bit floating point |
| 2     | bool | Boolean |
| 3     | string | UTF-8 string |
| 4     | nested | Nested vector/complex types |
| 5-255 | reserved | Reserved for future types |

### Storage Modes

| Value | Mode | Description |
|-------|------|-------------|
| 0     | dense | Contiguous memory array |
| 1     | sparse_coo | COOrdinate format |
| 2     | sparse_csr | Compressed Sparse Row |
| 3     | generator | Generator-backed (algorithm stored) |
| 4     | memoized | Cached lazy evaluation |
| 5-255 | reserved | Reserved for future modes |

### Metadata Flags

```
Bit | Description
----|-------------
0   | Compressed (LZ4)
1   | Encrypted
2   | Checksummed (SHA256)
3   | Big-endian byte order
4-7 | Compression level (0-15)
8-15| Reserved
```

### Data Section Layout

#### Dense Storage
```
[element_0, element_1, ..., element_n]
```
- Elements stored contiguously in native byte order
- Size = element_count * element_size

#### Sparse COO Storage
```
[index_section][value_section]
```
- Index section: array of [dim0, dim1, ..., dimN] coordinates
- Value section: array of element values
- Index size = nnz * dimensionality * 8 bytes
- Value size = nnz * element_size

#### Generator Storage
```
[algorithm_length][algorithm_data][parameters]
```
- Algorithm stored as serialized bytecode or AST
- Parameters stored as name-value pairs
- Enables "infinite storage" - store algorithm, not data

---

## Text Format (.xt) Specification

### Syntax Grammar

```
vector ::= scalar | array | object | generator

scalar ::= number | string | boolean | null

array ::= "[" [vector] ("," [vector])* "]"

object ::= "{" string ":" vector ("," string ":" vector)* "}"

generator ::= "gen(" function "," bounds ")"

function ::= identifier "(" [identifier] ("," [identifier])* ")"

bounds ::= "start:" number "," "end:" number
```

### Examples

#### Scalar Vector (0D)
```xt
42
```

#### 1D Vector
```xt
[1, 2, 3, 4, 5]
```

#### 2D Matrix
```xt
[[1, 2, 3],
 [4, 5, 6]]
```

#### Generator Vector
```xt
gen(n => n * 2, start: 0, end: 100)
```

#### Sparse Vector
```xt
sparse_coo(indices: [[0,0], [1,1]], values: [1, 2], shape: [10, 10])
```

#### Nested Structure
```xt
{
  "name": "vector_data",
  "values": [1, 2, 3, 4, 5],
  "metadata": {
    "created": "2025-01-01",
    "version": "1.0"
  }
}
```

---

## Graphics Format (.xg) Specification

### SVG-like Vector Graphics for Code

VEXL programs and data can be represented as vector graphics using an SVG-inspired format. This enables visual programming where code is edited as diagrams.

### Basic Structure

```xml
<vexl-program version="1.0">
  <!-- Functions as boxes -->
  <function name="add" x="100" y="100" width="120" height="80">
    <input name="a" type="i64" position="top"/>
    <input name="b" type="i64" position="top"/>
    <output type="i64" position="bottom"/>
    <body>
      <binop op="+" left="a" right="b"/>
    </body>
  </function>

  <!-- Pipelines as connected flows -->
  <pipeline x="300" y="100">
    <stage name="map" function="double"/>
    <stage name="filter" function="is_even"/>
    <stage name="reduce" function="sum"/>
    <connections>
      <connect from="map" to="filter"/>
      <connect from="filter" to="reduce"/>
    </connections>
  </pipeline>

  <!-- Vectors as visual arrays -->
  <vector name="data" x="50" y="300">
    <element value="1" index="0"/>
    <element value="2" index="1"/>
    <element value="3" index="2"/>
    <!-- ... -->
  </vector>
</vexl-program>
```

### Visual Elements

#### Functions
- Represented as rounded rectangles
- Inputs/outputs as connection points
- Color-coded by type/effect

#### Vectors
- Horizontal bars for 1D vectors
- Grids for 2D+ vectors
- Color intensity represents magnitude

#### Generators
- Infinite scrolls with pattern visualization
- Lazy evaluation indicators

#### Control Flow
- Branching paths with decision diamonds
- Loop constructs as circular arrows

---

## Archive Format (.xa) Specification

### Multi-Vector Container Format

Archives contain multiple vectors with shared metadata and compression.

### Header Structure

```
Magic: "VEXLA" (8 bytes)
Version: u32
Entry Count: u32
Total Size: u64
Compression: u8
Encryption: u8
Metadata Offset: u64
Data Offset: u64
```

### Entry Table

```
Name Length: u32
Name: [u8; name_length]
Vector Offset: u64
Vector Size: u64
Type Tag: u8
Dimensionality: u8
Shape: [u64; dimensionality]
```

### Compression Support

- LZ4: Fast compression/decompression
- ZSTD: High compression ratio
- GZIP: Compatible with existing tools
- None: Uncompressed for speed

---

## Stream Format (.xs) Specification

### Real-time Vector Streaming Protocol

Designed for network protocols and live data feeds.

### Message Structure

```
Message Type: u8 (0=data, 1=control, 2=error)
Stream ID: u32
Sequence: u64
Timestamp: u64
Payload Length: u32
Payload: [u8; payload_length]
Checksum: u32
```

### Message Types

#### Data Messages
- Vector data chunks
- Incremental updates
- Delta compression

#### Control Messages
- Stream initialization
- Parameter negotiation
- Flow control

#### Error Messages
- Transmission errors
- Protocol violations
- Recovery requests

### Streaming Operations

- **Append**: Add elements to vector
- **Update**: Modify existing elements
- **Delete**: Remove elements/ranges
- **Query**: Request vector slices
- **Subscribe**: Real-time update notifications

---

## Implementation Architecture

### Core Traits

```rust
pub trait VectorSerialize {
    fn to_xd(&self) -> Result<Vec<u8>>;
    fn from_xd(data: &[u8]) -> Result<Self>;

    fn to_xt(&self) -> Result<String>;
    fn from_xt(text: &str) -> Result<Self>;

    fn to_xg(&self) -> Result<String>;
    fn from_xg(xml: &str) -> Result<Self>;
}

pub trait VectorArchive {
    fn create_archive(vectors: HashMap<String, &dyn VectorSerialize>) -> Result<Vec<u8>>;
    fn read_archive(data: &[u8]) -> Result<HashMap<String, Box<dyn VectorSerialize>>>;
}

pub trait VectorStream {
    fn start_stream(&mut self, config: StreamConfig) -> Result<StreamHandle>;
    fn send_vector(&mut self, handle: &StreamHandle, vector: &dyn VectorSerialize) -> Result<()>;
    fn receive_vector(&mut self, handle: &StreamHandle) -> Result<Box<dyn VectorSerialize>>;
}
```

### Serialization Pipeline

1. **Type Analysis**: Determine vector structure and types
2. **Format Selection**: Choose optimal format based on use case
3. **Encoding**: Convert to target format with compression/metadata
4. **Validation**: Verify round-trip compatibility
5. **Storage**: Write to file/network/stream

### Performance Characteristics

| Format | Read Speed | Write Speed | Compression | Size |
|--------|------------|-------------|-------------|------|
| .xd    | Very Fast  | Very Fast  | Optional   | Small |
| .xt    | Slow       | Slow       | None       | Large |
| .xg    | Medium     | Medium     | Optional   | Medium |
| .xa    | Fast       | Fast       | Excellent  | Small |
| .xs    | Real-time  | Real-time  | Adaptive   | Variable |

---

## Security Considerations

### Encryption
- AES-256 for sensitive data
- Key derivation from passwords
- Hardware security module integration

### Integrity
- SHA256 checksums for all formats
- Merkle trees for large archives
- Digital signatures for authenticity

### Access Control
- Format-level permissions
- Encryption key management
- Audit logging for sensitive operations

---

## Future Extensions

### Planned Formats
- .xh: Vector HTML (web integration)
- .xj: Vector JSON (JavaScript interop)
- .xp: Vector Protocol Buffers (cross-language)

### Advanced Features
- Schema evolution support
- Zero-copy deserialization
- Hardware-accelerated compression
- Distributed vector storage

---

This specification enables VEXL's "everything is a vector" paradigm, providing a complete serialization ecosystem for the universal vector computing model.
