//! Foreign Function Interface (FFI) for the VEXL Runtime
//!
//! This module exposes C-ABI compatible functions that can be called
//! from LLVM-generated code.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::fs;
use std::io::{self};
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════
// I/O FUNCTIONS
// ═══════════════════════════════════════════════════════════

/// Print an integer (i64) to stdout
///
/// Called from LLVM as: call void @vexl_print_int(i64 %val)
#[no_mangle]
pub extern "C" fn vexl_print_int(n: i64) {
    println!("{}", n);
}

/// Print a float (f64) to stdout
#[no_mangle]
pub extern "C" fn vexl_print_float(x: f64) {
    println!("{}", x);
}

/// Print a string to stdout
///
/// Note: Expects a standard C-string (null-terminated) for now,
/// or we could pass length. Let's support length-based strings (VEXL style).
///
/// Signature: void vexl_print_string(i8* ptr, i64 len)
#[no_mangle]
pub extern "C" fn vexl_print_string(ptr: *const u8, len: u64) {
    if ptr.is_null() {
        println!("(null)");
        return;
    }

    // Safety: We assume the compiler generates valid pointers and lengths
    let slice = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let s = String::from_utf8_lossy(slice);
    println!("{}", s);
}

/// Print a null-terminated C-string (legacy/debug support)
#[no_mangle]
pub extern "C" fn vexl_print_cstr(ptr: *const c_char) {
    if ptr.is_null() {
        return;
    }

    let c_str = unsafe { CStr::from_ptr(ptr) };
    if let Ok(s) = c_str.to_str() {
        println!("{}", s);
    }
}

/// Read a line from stdin
/// Returns: pointer to allocated string (null if error)
/// The caller is responsible for freeing the memory
#[no_mangle]
pub extern "C" fn vexl_read_line() -> *mut u8 {
    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => {
            // Remove trailing newline
            let input = input.trim_end();
            // Allocate and return as C string
            let c_str = std::ffi::CString::new(input).unwrap_or_default();
            c_str.into_raw() as *mut u8
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Read entire file into memory
/// Returns: pointer to allocated buffer, length stored in len_out
/// Returns null on error
#[no_mangle]
pub extern "C" fn vexl_read_file(path_ptr: *const u8, path_len: u64, len_out: *mut u64) -> *mut u8 {
    if path_ptr.is_null() {
        unsafe { *len_out = 0; }
        return std::ptr::null_mut();
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path_ptr, path_len as usize) };
    let path = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => {
            unsafe { *len_out = 0; }
            return std::ptr::null_mut();
        }
    };

    match fs::read(path) {
        Ok(data) => {
            let len = data.len();
            unsafe { *len_out = len as u64; }

            // Allocate buffer and copy data
            let layout = std::alloc::Layout::from_size_align(len, 1).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return std::ptr::null_mut();
            }

            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len); }

            // Register with GC
            crate::gc::gc_register(ptr, layout.size());

            ptr
        }
        Err(_) => {
            unsafe { *len_out = 0; }
            std::ptr::null_mut()
        }
    }
}

/// Write buffer to file
/// Returns: 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn vexl_write_file(path_ptr: *const u8, path_len: u64, data_ptr: *const u8, data_len: u64) -> i32 {
    if path_ptr.is_null() || data_ptr.is_null() {
        return -1;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path_ptr, path_len as usize) };
    let path = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };

    match fs::write(path, data_slice) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Check if file exists
/// Returns: 1 if exists, 0 if not, -1 on error
#[no_mangle]
pub extern "C" fn vexl_file_exists(path_ptr: *const u8, path_len: u64) -> i32 {
    if path_ptr.is_null() {
        return -1;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path_ptr, path_len as usize) };
    let path = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match fs::metadata(path) {
        Ok(_) => 1,
        Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
        Err(_) => -1,
    }
}

/// Get file size
/// Returns: file size in bytes, or -1 on error
#[no_mangle]
pub extern "C" fn vexl_file_size(path_ptr: *const u8, path_len: u64) -> i64 {
    if path_ptr.is_null() {
        return -1;
    }

    let path_slice = unsafe { std::slice::from_raw_parts(path_ptr, path_len as usize) };
    let path = match std::str::from_utf8(path_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match fs::metadata(path) {
        Ok(metadata) => metadata.len() as i64,
        Err(_) => -1,
    }
}

// ═══════════════════════════════════════════════════════════
// SYSTEM FUNCTIONS
// ═══════════════════════════════════════════════════════════

/// Get current timestamp (Unix epoch in seconds)
#[no_mangle]
pub extern "C" fn vexl_current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Get current timestamp with nanosecond precision
#[no_mangle]
pub extern "C" fn vexl_current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Sleep for specified milliseconds
#[no_mangle]
pub extern "C" fn vexl_sleep_ms(milliseconds: u64) {
    std::thread::sleep(std::time::Duration::from_millis(milliseconds));
}

/// Get environment variable
/// Returns: pointer to allocated string, or null if not found
#[no_mangle]
pub extern "C" fn vexl_getenv(var_ptr: *const u8, var_len: u64) -> *mut u8 {
    if var_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let var_slice = unsafe { std::slice::from_raw_parts(var_ptr, var_len as usize) };
    let var_name = match std::str::from_utf8(var_slice) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match std::env::var(var_name) {
        Ok(value) => {
            let c_str = std::ffi::CString::new(value).unwrap_or_default();
            c_str.into_raw() as *mut u8
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set environment variable
/// Returns: 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn vexl_setenv(key_ptr: *const u8, key_len: u64, value_ptr: *const u8, value_len: u64) -> i32 {
    if key_ptr.is_null() || value_ptr.is_null() {
        return -1;
    }

    let key_slice = unsafe { std::slice::from_raw_parts(key_ptr, key_len as usize) };
    let value_slice = unsafe { std::slice::from_raw_parts(value_ptr, value_len as usize) };

    let key = match std::str::from_utf8(key_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let value = match std::str::from_utf8(value_slice) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match std::env::set_var(key, value) {
        () => 0,
    }
}

/// Get command line arguments as vector
/// Returns: pointer to vector of strings
#[no_mangle]
pub extern "C" fn vexl_get_args() -> *mut crate::vector::Vector {
    let args: Vec<String> = std::env::args().collect();

    // Convert to vector of C strings
    let c_strings: Vec<std::ffi::CString> = args.into_iter()
        .map(|s| std::ffi::CString::new(s).unwrap_or_default())
        .collect();

    // Create vector of pointers
    let ptrs: Vec<*mut u8> = c_strings.into_iter()
        .map(|cs| cs.into_raw() as *mut u8)
        .collect();

    // Use FFI function to create vector from array
    // Note: This assumes pointer-sized elements (8 bytes on 64-bit)
    let count = ptrs.len() as u64;
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate data array
    let data_layout = std::alloc::Layout::from_size_align((count * 8) as usize, 8).unwrap();
    let data_ptr = unsafe { std::alloc::alloc(data_layout) };

    // Copy pointers
    unsafe {
        std::ptr::copy_nonoverlapping(
            ptrs.as_ptr() as *const u8,
            data_ptr,
            (count * 8) as usize
        );
    }

    // Create vector (type tag: 5 = pointer/vector)
    unsafe { crate::vector::Vector::from_raw_parts(5, 1, count, data_ptr) }
}

/// Exit the program with status code
#[no_mangle]
pub extern "C" fn vexl_exit(code: i32) -> ! {
    std::process::exit(code);
}

/// Get process ID
#[no_mangle]
pub extern "C" fn vexl_getpid() -> i32 {
    std::process::id() as i32
}

/// Generate random number (0.0 to 1.0)
#[no_mangle]
pub extern "C" fn vexl_random() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple PRNG using current time as seed
    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    let seed = hasher.finish();

    // Linear congruential generator
    let a = 1664525u64;
    let c = 1013904223u64;
    let m = 4294967296u64;

    let random_int = (seed.wrapping_mul(a).wrapping_add(c)) % m;
    random_int as f64 / m as f64
}

// ═══════════════════════════════════════════════════════════
// MEMORY MANAGEMENT HELPERS
// ═══════════════════════════════════════════════════════════

/// Allocate memory with GC registration
#[no_mangle]
pub extern "C" fn vexl_alloc(size: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };

    if !ptr.is_null() {
        crate::gc::gc_register(ptr, size);
    }

    ptr
}

/// Free memory (with GC unregistration)
#[no_mangle]
pub extern "C" fn vexl_free(ptr: *mut u8) {
    if !ptr.is_null() {
        crate::gc::gc_unregister(ptr);
        unsafe {
            let layout = std::alloc::Layout::from_size_align(0, 8).unwrap(); // Size unknown, but we can still dealloc
            std::alloc::dealloc(ptr, layout);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// STRING UTILITIES
// ═══════════════════════════════════════════════════════════

/// Compare two strings (length-based)
/// Returns: 0 if equal, negative if s1 < s2, positive if s1 > s2
#[no_mangle]
pub extern "C" fn vexl_string_compare(
    s1_ptr: *const u8, s1_len: u64,
    s2_ptr: *const u8, s2_len: u64
) -> i32 {
    if s1_ptr.is_null() || s2_ptr.is_null() {
        return if s1_ptr == s2_ptr { 0 } else { -1 };
    }

    let s1 = unsafe { std::slice::from_raw_parts(s1_ptr, s1_len as usize) };
    let s2 = unsafe { std::slice::from_raw_parts(s2_ptr, s2_len as usize) };

    match (std::str::from_utf8(s1), std::str::from_utf8(s2)) {
        (Ok(s1_str), Ok(s2_str)) => s1_str.cmp(s2_str) as i32,
        _ => s1.cmp(s2) as i32,
    }
}

/// Get substring
/// Returns: pointer to allocated substring
#[no_mangle]
pub extern "C" fn vexl_string_substring(
    s_ptr: *const u8, s_len: u64,
    start: u64, len: u64
) -> *mut u8 {
    if s_ptr.is_null() || start >= s_len {
        return std::ptr::null_mut();
    }

    let slice = unsafe { std::slice::from_raw_parts(s_ptr, s_len as usize) };
    let substr_len = std::cmp::min(len as usize, s_len as usize - start as usize);
    let substr = &slice[start as usize..start as usize + substr_len];

    let c_str = std::ffi::CString::new(substr).unwrap_or_default();
    c_str.into_raw() as *mut u8
}
