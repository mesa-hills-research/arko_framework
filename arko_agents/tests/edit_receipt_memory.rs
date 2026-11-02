//! Making an edit's receipt takes memory in proportion to the receipt, not
//! to the change, so replacing a large file whole allocates nothing larger
//! than the file itself.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arko_agents::TaskContext;
use arko_agents::builtins::EditFileTool;
use arko_agents::tool::{Tool, ToolReceipt};

/// The system's allocator, which notes the largest allocation while it
/// measures.
struct Measured;

static MEASURING: AtomicBool = AtomicBool::new(false);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Measured {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.load(Ordering::Relaxed) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if MEASURING.load(Ordering::Relaxed) {
            LARGEST.fetch_max(size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Measured = Measured;

#[tokio::test]
async fn replacing_a_large_file_whole_allocates_nothing_larger_than_the_file() {
    let folder = std::env::temp_dir().join(format!("arko-receipt-{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    // A million short lines, two megabytes.
    let file = "a\n".repeat(1_000_000);
    std::fs::write(folder.join("large.txt"), &file).unwrap();
    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(
        serde_json::json!({"path": "large.txt", "old_string": file, "new_string": "b\n"}),
    )
    .unwrap();
    let context = TaskContext::new(uuid::Uuid::new_v4(), "tester", "", &folder);
    LARGEST.store(0, Ordering::Relaxed);
    MEASURING.store(true, Ordering::Relaxed);
    let output = EditFileTool
        .execute_with_output(&context, parameters)
        .await
        .unwrap();
    MEASURING.store(false, Ordering::Relaxed);
    let largest = LARGEST.load(Ordering::Relaxed);
    std::fs::remove_dir_all(&folder).unwrap();
    let Some(ToolReceipt::Edit(receipt)) = output.receipt else {
        panic!("no receipt");
    };
    assert!(receipt.truncated);
    // The tool holds the file's text, its new text and what it writes, each
    // about the file's size.
    assert!(
        largest <= 2 * file.len(),
        "{largest} bytes at once for a {} byte file",
        file.len()
    );
}
