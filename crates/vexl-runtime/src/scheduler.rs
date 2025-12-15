//! Cooperative Scheduler - Load-Balancing Parallel Execution
//!
//! This scheduler uses a cooperative work distribution strategy where idle threads
//! help busy threads by taking tasks from their queues. This ensures optimal CPU
//! utilization and fair work distribution across all available cores.

#![allow(static_mut_refs)]

use crossbeam_deque::{Injector, Stealer, Worker};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

type Task = Box<dyn FnOnce() + Send + 'static>;

pub struct CooperativeScheduler {
    injector: Arc<Injector<Task>>,
    _stealers: Vec<Stealer<Task>>,
    workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    _num_threads: usize,
}

impl CooperativeScheduler {
    /// Create a new cooperative scheduler with specified number of threads
    pub fn new(num_threads: usize) -> Self {
        let injector = Arc::new(Injector::new());
        let mut stealers = Vec::new();
        let mut workers = Vec::new();
        let shutdown = Arc::new(AtomicBool::new(false));

        for _ in 0..num_threads {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            
            let injector = Arc::clone(&injector);
            let worker_stealers = stealers.iter().map(|s| s.clone()).collect::<Vec<_>>();
            let shutdown = Arc::clone(&shutdown);
            
            let handle = thread::spawn(move || {
                cooperative_worker_loop(worker, injector, worker_stealers, shutdown);
            });
            
            workers.push(handle);
        }

        Self {
            injector,
            _stealers: stealers,
            workers,
            shutdown,
            _num_threads: num_threads,
        }
    }

    /// Submit a task to the scheduler
    pub fn submit<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.injector.push(Box::new(task));
    }

    /// Shutdown the scheduler and wait for all workers to finish
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for handle in self.workers {
            handle.join().ok();
        }
    }
}

impl Default for CooperativeScheduler {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::new(num_cpus)
    }
}

/// Worker loop that cooperatively distributes work
fn cooperative_worker_loop(
    worker: Worker<Task>,
    injector: Arc<Injector<Task>>,
    stealers: Vec<Stealer<Task>>,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::SeqCst) {
        // Try to get task from local queue first
        let task = worker.pop()
            // If local queue is empty, try global injector
            .or_else(|| loop {
                match injector.steal_batch_and_pop(&worker) {
                    crossbeam_deque::Steal::Success(t) => break Some(t),
                    crossbeam_deque::Steal::Empty => break None,
                    crossbeam_deque::Steal::Retry => continue,
                }
            })
            // If still no task, help other workers by taking from their queues
            .or_else(|| {
                stealers.iter().map(|s| loop {
                    match s.steal_batch_and_pop(&worker) {
                        crossbeam_deque::Steal::Success(t) => break Some(t),
                        crossbeam_deque::Steal::Empty => break None,
                        crossbeam_deque::Steal::Retry => continue,
                    }
                }).find_map(|t| t)
            });

        if let Some(task) = task {
            task();
        } else {
            // No work available, yield to avoid busy-waiting
            thread::yield_now();
        }
    }
}

/// Global scheduler instance
static mut GLOBAL_SCHEDULER: Option<CooperativeScheduler> = None;

/// Initialize the global thread pool
pub fn init_thread_pool() {
    unsafe {
        if GLOBAL_SCHEDULER.is_none() {
            GLOBAL_SCHEDULER = Some(CooperativeScheduler::default());
        }
    }
}

/// Shutdown the global thread pool
pub fn shutdown_thread_pool() {
    unsafe {
        if let Some(scheduler) = GLOBAL_SCHEDULER.take() {
            scheduler.shutdown();
        }
    }
}

/// Get reference to global scheduler
pub fn global_scheduler() -> &'static CooperativeScheduler {
    unsafe {
        GLOBAL_SCHEDULER.as_ref().expect("Thread pool not initialized")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_cooperative_scheduler() {
        let scheduler = CooperativeScheduler::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..100 {
            let counter = Arc::clone(&counter);
            scheduler.submit(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Give tasks time to complete
        thread::sleep(std::time::Duration::from_millis(100));
        scheduler.shutdown();

        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }
}
