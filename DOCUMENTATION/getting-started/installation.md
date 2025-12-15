# Installation Guide

> **Installing VEXL: From Zero to Running Your First Program**

## What You'll Learn

In this guide, you'll learn how to install VEXL on your computer and run your first program. No prior programming experience required!

## Prerequisites

Before installing VEXL, you need Rust installed on your computer. Rust is the language VEXL is written in, and it includes everything we need to build and run VEXL programs.

### Installing Rust

**Step 1: Download Rust**

Visit the official Rust website and run the installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Step 2: Restart Your Terminal**

Close and reopen your terminal (or command prompt) so the changes take effect.

**Step 3: Verify Installation**

Open a new terminal and type:

```bash
rustc --version
```

You should see something like:
```
rustc 1.70.0 (ec8a8a0ca 2023-05-25)
```

## Installing VEXL

Now that Rust is installed, let's install VEXL:

### Option 1: Build from Source (Recommended)

**Step 1: Clone the Repository**

```bash
git clone https://github.com/your-org/vexl-language.git
cd vexl-language
```

**Step 2: Build VEXL**

This will compile VEXL from source. It might take a few minutes the first time:

```bash
cargo build --release
```

**Step 3: Test the Installation**

```bash
./target/release/vexl --help
```

You should see the VEXL help message.

### Option 2: Use Pre-built Binary (When Available)

If pre-built binaries are available for your platform:

```bash
# Download the latest release
curl -L https://github.com/your-org/vexl-language/releases/latest/download/vexl -o vexl
chmod +x vexl

# Test it
./vexl --help
```

## Verifying Your Installation

Create a simple test file to verify everything works:

**Step 1: Create a Test File**

```bash
echo "let result = 2 + 2 * 3" > test.vexl
```

**Step 2: Run It**

```bash
./target/release/vexl run test.vexl
```

You should see output like:
```
Result: 8
```

## Troubleshooting

### Rust Installation Issues

**Problem:** `rustc: command not found`

**Solution:** Make sure Rust is in your PATH:
```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### VEXL Build Issues

**Problem:** `error: failed to compile`

**Solution:** Make sure you have the latest Rust:
```bash
rustup update
cargo clean
cargo build --release
```

### Permission Issues

**Problem:** `Permission denied`

**Solution:** Make the binary executable:
```bash
chmod +x ./target/release/vexl
```

## System Requirements

### Minimum Requirements
- **Operating System:** Linux, macOS, or Windows 10/11
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB for Rust + VEXL source
- **Processor:** Any 64-bit processor

### Recommended for Development
- **RAM:** 16GB or more
- **Disk Space:** 10GB free space
- **Processor:** Multi-core processor for faster compilation

## Platform-Specific Notes

### Linux
Most Linux distributions work out of the box. For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential
```

### macOS
You need Xcode Command Line Tools:
```bash
xcode-select --install
```

### Windows
On Windows, use Windows Subsystem for Linux (WSL) or install directly with:
```bash
# In PowerShell or Command Prompt
curl -sSf https://sh.rustup.rs | sh
```

## Development Tools (Optional)

For a better development experience, consider installing these tools:

### Cargo Watch
Automatically rebuilds when files change:
```bash
cargo install cargo-watch
```

### Cargo Audit
Checks for security vulnerabilities:
```bash
cargo install cargo-audit
```

### Editor Extensions
- **VS Code:** VEXL syntax highlighting (coming soon)
- **Vim/Neovim:** VEXL syntax files
- **Emacs:** VEXL mode

## What's Next?

Once VEXL is installed, you're ready to:

1. [Write Your First Program](first-program.md)
2. [Learn Basic Concepts](basic-concepts.md)
3. [Follow the Tutorial](tutorial.md)

## Getting Help

If you run into problems:

- **GitHub Issues:** [Report bugs](https://github.com/your-org/vexl-language/issues)
- **Discord:** [Join our community](https://discord.gg/vexl)
- **Forum:** [Ask questions](https://forum.vexl.dev)

---

**Next:** [Writing Your First Program](first-program.md)
