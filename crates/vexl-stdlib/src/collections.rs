//! Data Structures and Collections for VEXL
//!
//! Essential data structures including:
//! - HashMap and HashSet implementations
//! - Ordered maps (BTreeMap) and sets
//! - Graph structures (adjacency list, matrix)
//! - Priority queues and heaps
//! - Stack and queue implementations

use vexl_runtime::vector::Vector;
use vexl_runtime::context::{ExecutionContext, Value, Function};
use std::collections::{HashMap as StdHashMap, HashSet as StdHashSet, BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::rc::Rc;
use std::cell::RefCell;

/// HashMap implementation for VEXL
pub struct HashMap {
    inner: StdHashMap<i64, i64>, // Simplified key-value storage
}

impl HashMap {
    /// Create a new empty HashMap
    pub fn new() -> Self {
        Self {
            inner: StdHashMap::new(),
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: i64, value: i64) -> Option<i64> {
        self.inner.insert(key, value)
    }

    /// Get a value by key
    pub fn get(&self, key: i64) -> Option<i64> {
        self.inner.get(&key).copied()
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: i64) -> Option<i64> {
        self.inner.remove(&key)
    }

    /// Check if the map contains a key
    pub fn contains_key(&self, key: i64) -> bool {
        self.inner.contains_key(&key)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get all keys as a vector
    pub fn keys(&self) -> Vec<i64> {
        self.inner.keys().copied().collect()
    }

    /// Get all values as a vector
    pub fn values(&self) -> Vec<i64> {
        self.inner.values().copied().collect()
    }
}

/// HashSet implementation for VEXL
pub struct HashSet {
    inner: StdHashSet<i64>,
}

impl HashSet {
    /// Create a new empty HashSet
    pub fn new() -> Self {
        Self {
            inner: StdHashSet::new(),
        }
    }

    /// Insert a value
    pub fn insert(&mut self, value: i64) -> bool {
        self.inner.insert(value)
    }

    /// Remove a value
    pub fn remove(&mut self, value: i64) -> bool {
        self.inner.remove(&value)
    }

    /// Check if the set contains a value
    pub fn contains(&self, value: i64) -> bool {
        self.inner.contains(&value)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Convert to vector
    pub fn to_vec(&self) -> Vec<i64> {
        self.inner.iter().copied().collect()
    }

    /// Union with another set
    pub fn union(&self, other: &HashSet) -> HashSet {
        let mut result = self.inner.clone();
        result.extend(&other.inner);
        HashSet { inner: result }
    }

    /// Intersection with another set
    pub fn intersection(&self, other: &HashSet) -> HashSet {
        let result = self.inner.intersection(&other.inner).copied().collect();
        HashSet { inner: result }
    }

    /// Difference with another set
    pub fn difference(&self, other: &HashSet) -> HashSet {
        let result = self.inner.difference(&other.inner).copied().collect();
        HashSet { inner: result }
    }
}

/// Ordered map (BTreeMap) implementation
pub struct OrderedMap {
    inner: BTreeMap<i64, i64>,
}

impl OrderedMap {
    /// Create a new empty OrderedMap
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: i64, value: i64) -> Option<i64> {
        self.inner.insert(key, value)
    }

    /// Get a value by key
    pub fn get(&self, key: i64) -> Option<i64> {
        self.inner.get(&key).copied()
    }

    /// Get the first (smallest) key-value pair
    pub fn first(&self) -> Option<(i64, i64)> {
        self.inner.first_key_value().map(|(k, v)| (*k, *v))
    }

    /// Get the last (largest) key-value pair
    pub fn last(&self) -> Option<(i64, i64)> {
        self.inner.last_key_value().map(|(k, v)| (*k, *v))
    }

    /// Get all keys in order
    pub fn keys(&self) -> Vec<i64> {
        self.inner.keys().copied().collect()
    }

    /// Get all values in key order
    pub fn values(&self) -> Vec<i64> {
        self.inner.values().copied().collect()
    }

    /// Get range of key-value pairs
    pub fn range(&self, start: i64, end: i64) -> Vec<(i64, i64)> {
        self.inner.range(start..=end).map(|(k, v)| (*k, *v)).collect()
    }
}

/// Ordered set (BTreeSet) implementation
pub struct OrderedSet {
    inner: BTreeSet<i64>,
}

impl OrderedSet {
    /// Create a new empty OrderedSet
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    /// Insert a value
    pub fn insert(&mut self, value: i64) -> bool {
        self.inner.insert(value)
    }

    /// Remove a value
    pub fn remove(&mut self, value: i64) -> bool {
        self.inner.remove(&value)
    }

    /// Get the first (smallest) element
    pub fn first(&self) -> Option<i64> {
        self.inner.first().copied()
    }

    /// Get the last (largest) element
    pub fn last(&self) -> Option<i64> {
        self.inner.last().copied()
    }

    /// Convert to vector in order
    pub fn to_vec(&self) -> Vec<i64> {
        self.inner.iter().copied().collect()
    }

    /// Get elements in a range
    pub fn range(&self, start: i64, end: i64) -> Vec<i64> {
        self.inner.range(start..=end).copied().collect()
    }
}

/// Priority queue implementation using BinaryHeap
pub struct PriorityQueue {
    inner: BinaryHeap<i64>, // Max-heap by default
}

impl PriorityQueue {
    /// Create a new empty priority queue
    pub fn new() -> Self {
        Self {
            inner: BinaryHeap::new(),
        }
    }

    /// Push an element
    pub fn push(&mut self, value: i64) {
        self.inner.push(value);
    }

    /// Pop the highest priority element
    pub fn pop(&mut self) -> Option<i64> {
        self.inner.pop()
    }

    /// Peek at the highest priority element
    pub fn peek(&self) -> Option<i64> {
        self.inner.peek().copied()
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Stack implementation
pub struct Stack {
    inner: Vec<i64>,
}

impl Stack {
    /// Create a new empty stack
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
        }
    }

    /// Push an element onto the stack
    pub fn push(&mut self, value: i64) {
        self.inner.push(value);
    }

    /// Pop an element from the stack
    pub fn pop(&mut self) -> Option<i64> {
        self.inner.pop()
    }

    /// Peek at the top element
    pub fn peek(&self) -> Option<i64> {
        self.inner.last().copied()
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear the stack
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Queue implementation
pub struct Queue {
    inner: VecDeque<i64>,
}

impl Queue {
    /// Create a new empty queue
    pub fn new() -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }

    /// Add an element to the back of the queue
    pub fn enqueue(&mut self, value: i64) {
        self.inner.push_back(value);
    }

    /// Remove an element from the front of the queue
    pub fn dequeue(&mut self) -> Option<i64> {
        self.inner.pop_front()
    }

    /// Peek at the front element
    pub fn peek(&self) -> Option<i64> {
        self.inner.front().copied()
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear the queue
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Graph representation using adjacency list
pub struct Graph {
    /// Adjacency list: node -> list of (neighbor, weight)
    adj_list: StdHashMap<i64, Vec<(i64, i64)>>,
    /// Whether this is a directed graph
    directed: bool,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(directed: bool) -> Self {
        Self {
            adj_list: StdHashMap::new(),
            directed,
        }
    }

    /// Add a vertex
    pub fn add_vertex(&mut self, vertex: i64) {
        self.adj_list.entry(vertex).or_insert_with(Vec::new);
    }

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, from: i64, to: i64, weight: i64) {
        self.adj_list.entry(from).or_insert_with(Vec::new).push((to, weight));

        if !self.directed {
            self.adj_list.entry(to).or_insert_with(Vec::new).push((from, weight));
        }
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: i64) -> Vec<i64> {
        self.adj_list.get(&vertex)
            .map(|neighbors| neighbors.iter().map(|(v, _)| *v).collect())
            .unwrap_or_default()
    }

    /// Get edges from a vertex (with weights)
    pub fn edges(&self, vertex: i64) -> Vec<(i64, i64)> {
        self.adj_list.get(&vertex).cloned().unwrap_or_default()
    }

    /// Check if vertex exists
    pub fn has_vertex(&self, vertex: i64) -> bool {
        self.adj_list.contains_key(&vertex)
    }

    /// Get all vertices
    pub fn vertices(&self) -> Vec<i64> {
        self.adj_list.keys().copied().collect()
    }

    /// Get number of vertices
    pub fn vertex_count(&self) -> usize {
        self.adj_list.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        let mut count = 0;
        for neighbors in self.adj_list.values() {
            count += neighbors.len();
        }
        if !self.directed {
            count /= 2; // Each edge is counted twice in undirected graph
        }
        count
    }

    /// Breadth-first search
    pub fn bfs(&self, start: i64) -> Vec<i64> {
        let mut visited = StdHashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(vertex) = queue.pop_front() {
            result.push(vertex);

            for neighbor in self.neighbors(vertex) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        result
    }

    /// Depth-first search
    pub fn dfs(&self, start: i64) -> Vec<i64> {
        let mut visited = StdHashSet::new();
        let mut result = Vec::new();

        self.dfs_recursive(start, &mut visited, &mut result);

        result
    }

    fn dfs_recursive(&self, vertex: i64, visited: &mut StdHashSet<i64>, result: &mut Vec<i64>) {
        visited.insert(vertex);
        result.push(vertex);

        for neighbor in self.neighbors(vertex) {
            if visited.insert(neighbor) {
                self.dfs_recursive(neighbor, visited, result);
            }
        }
    }
}

/// Register collections operations with the execution context
pub fn register_collections_ops(context: &mut ExecutionContext) {
    // HashMap operations
    context.register_function(Function::Native {
        name: "hashmap_new".to_string(),
        arg_count: 0,
        func: Rc::new(|_args| {
            let map = Rc::new(RefCell::new(HashMap::new()));
            // In practice, we'd need to store this in the context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    context.register_function(Function::Native {
        name: "hashmap_insert".to_string(),
        arg_count: 3,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 3 {
                return Err("hashmap_insert requires 3 arguments".to_string());
            }
            // Implementation would require storing hashmaps in context
            Ok(Value::Unit)
        }),
    });

    // HashSet operations
    context.register_function(Function::Native {
        name: "hashset_new".to_string(),
        arg_count: 0,
        func: Rc::new(|_args| {
            let set = Rc::new(RefCell::new(HashSet::new()));
            // In practice, we'd need to store this in the context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    // Stack operations
    context.register_function(Function::Native {
        name: "stack_new".to_string(),
        arg_count: 0,
        func: Rc::new(|_args| {
            let stack = Rc::new(RefCell::new(Stack::new()));
            // In practice, we'd need to store this in the context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    context.register_function(Function::Native {
        name: "stack_push".to_string(),
        arg_count: 2,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 2 {
                return Err("stack_push requires 2 arguments".to_string());
            }
            // Implementation would require retrieving stack from context
            Ok(Value::Unit)
        }),
    });

    context.register_function(Function::Native {
        name: "stack_pop".to_string(),
        arg_count: 1,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 1 {
                return Err("stack_pop requires 1 argument".to_string());
            }
            // Implementation would require retrieving stack from context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    // Queue operations
    context.register_function(Function::Native {
        name: "queue_new".to_string(),
        arg_count: 0,
        func: Rc::new(|_args| {
            let queue = Rc::new(RefCell::new(Queue::new()));
            // In practice, we'd need to store this in the context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    // Priority queue operations
    context.register_function(Function::Native {
        name: "priority_queue_new".to_string(),
        arg_count: 0,
        func: Rc::new(|_args| {
            let pq = Rc::new(RefCell::new(PriorityQueue::new()));
            // In practice, we'd need to store this in the context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });

    context.register_function(Function::Native {
        name: "priority_queue_push".to_string(),
        arg_count: 2,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 2 {
                return Err("priority_queue_push requires 2 arguments".to_string());
            }
            // Implementation would require retrieving priority queue from context
            Ok(Value::Unit)
        }),
    });

    context.register_function(Function::Native {
        name: "priority_queue_pop".to_string(),
        arg_count: 1,
        func: Rc::new(|args: &[Value]| {
            if args.len() != 1 {
                return Err("priority_queue_pop requires 1 argument".to_string());
            }
            // Implementation would require retrieving priority queue from context
            Ok(Value::Integer(0)) // Placeholder
        }),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap_operations() {
        let mut map = HashMap::new();

        assert_eq!(map.insert(1, 10), None);
        assert_eq!(map.insert(2, 20), None);
        assert_eq!(map.insert(1, 30), Some(10)); // Overwrite

        assert_eq!(map.get(1), Some(30));
        assert_eq!(map.get(2), Some(20));
        assert_eq!(map.get(3), None);

        assert!(map.contains_key(1));
        assert!(!map.contains_key(3));

        assert_eq!(map.len(), 2);

        let keys = map.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
    }

    #[test]
    fn test_hashset_operations() {
        let mut set = HashSet::new();

        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(!set.insert(1)); // Already exists

        assert!(set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(3));

        assert_eq!(set.len(), 2);

        let vec = set.to_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.contains(&1));
        assert!(vec.contains(&2));
    }

    #[test]
    fn test_stack_operations() {
        let mut stack = Stack::new();

        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.peek(), Some(3));

        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);

        assert!(stack.is_empty());
    }

    #[test]
    fn test_queue_operations() {
        let mut queue = Queue::new();

        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.peek(), Some(1));

        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_queue_operations() {
        let mut pq = PriorityQueue::new();

        pq.push(3);
        pq.push(1);
        pq.push(4);
        pq.push(2);

        assert_eq!(pq.len(), 4);
        assert_eq!(pq.peek(), Some(4)); // Max-heap

        assert_eq!(pq.pop(), Some(4));
        assert_eq!(pq.pop(), Some(3));
        assert_eq!(pq.pop(), Some(2));
        assert_eq!(pq.pop(), Some(1));
        assert_eq!(pq.pop(), None);

        assert!(pq.is_empty());
    }

    #[test]
    fn test_graph_operations() {
        let mut graph = Graph::new(false); // Undirected

        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(1, 2, 5);
        graph.add_edge(2, 3, 3);
        graph.add_edge(1, 3, 7);

        assert!(graph.has_vertex(1));
        assert!(!graph.has_vertex(4));

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3); // Undirected, so each edge counted once

        let neighbors_1 = graph.neighbors(1);
        assert_eq!(neighbors_1.len(), 2);
        assert!(neighbors_1.contains(&2));
        assert!(neighbors_1.contains(&3));
    }

    #[test]
    fn test_graph_bfs() {
        let mut graph = Graph::new(false);

        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);

        let bfs_result = graph.bfs(1);
        assert!(!bfs_result.is_empty());
        assert_eq!(bfs_result[0], 1); // Start with root
    }

    #[test]
    fn test_register_collections_ops() {
        let mut context = ExecutionContext::new();
        register_collections_ops(&mut context);

        // Check that functions were registered
        assert!(context.call_function("hashmap_new", &[]).is_ok());
        assert!(context.call_function("stack_new", &[]).is_ok());
        assert!(context.call_function("nonexistent", &[]).is_err());
    }
}
