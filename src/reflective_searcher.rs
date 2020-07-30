// Implements a reflective searcher data structure.
//
// This data structure supports the algorithm in my research notebook.

// I need to start by creating an API and unit tests.

use std::collections::BTreeMap;
use crate::Cost;

#[derive(Debug,Clone)]
pub struct MultiStageQueue<C:Cost,T> {
    backtrack_stack: Vec<T>,
    holding_area: Vec<T>,
    prioritized: BTreeMap<C,Vec<T>>
}

impl <C:Cost,T> MultiStageQueue<C,T> {
    pub fn new() -> Self {
        MultiStageQueue {backtrack_stack: Vec::new(), holding_area: Vec::new(),
        prioritized: BTreeMap::new()}
    }

    pub fn len(&self) -> usize {
        let num_in_heap: usize = self.prioritized.iter().map(|(_,v)| v.len()).sum();
        self.backtrack_stack.len() + self.holding_area.len() + num_in_heap
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, obj: T) {
        self.backtrack_stack.push(obj);
    }

    pub fn remove(&mut self) -> Option<T> {
        if self.backtrack_stack.is_empty() {
            if self.holding_area.is_empty() {
                match self.prioritized.first_entry() {
                    None => None,
                    Some(mut entry) => {
                        let result = entry.get_mut().pop().unwrap();
                        if entry.get().is_empty() {
                            entry.remove_entry();
                        }
                        Some(result)
                    }
                }
            } else {
                self.holding_area.pop()
            }
        } else {
            self.backtrack_stack.pop()
        }
    }

    pub fn to_holding(&mut self) {
        let mut moving: Vec<T> = self.backtrack_stack.drain(..).collect();
        self.holding_area.append(&mut moving);
    }

    pub fn to_heap(&mut self, cost: C) {
        match self.prioritized.get_mut(&cost) {
            None => {self.prioritized.insert(cost, self.backtrack_stack.drain(..).collect());},
            Some(v) => self.backtrack_stack.drain(..).for_each(|t| v.push(t))
        }
    }

    pub fn to_heap_bfs(&mut self) {
        use num_traits::identities::{zero, one};
        let mut priority = self.prioritized.last_key_value()
            .map(|pair| *(pair.0))
            .unwrap_or(zero());

        let mut items: Vec<T> = self.backtrack_stack.drain(..).collect();
        items.drain(..).for_each(|item| {
            priority = priority + one();
            self.add_to_heap(item, priority)});
    }

    fn add_to_heap(&mut self, item: T, cost: C) {
        match self.prioritized.get_mut(&cost) {
            None => {self.prioritized.insert(cost, vec![item]);},
            Some(v) => v.push(item)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::reflective_searcher::MultiStageQueue;

    #[test]
    fn test1() {
        let mut m = MultiStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);
        assert_eq!(m.len(), 3);
        let three = m.remove().unwrap();
        assert_eq!(three, 3);
        assert_eq!(m.len(), 2);
        m.to_holding();
        m.insert(30);
        m.insert(31);
        m.insert(32);
        assert_eq!(m.len(), 5);
        let three_2 = m.remove().unwrap();
        assert_eq!(three_2, 32);
        assert_eq!(m.len(), 4);
        m.to_heap(3);
        let two = m.remove().unwrap();
        assert_eq!(two, 2);
        assert_eq!(m.len(), 3);
        m.insert(20);
        m.insert(21);
        m.insert(22);
        assert_eq!(m.len(), 6);
        let two_2 = m.remove().unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(two_2, 22);
        m.to_heap(2);
        let one = m.remove().unwrap();
        assert_eq!(one, 1);
        assert_eq!(m.len(), 4);
        m.insert(10);
        m.insert(11);
        m.insert(12);
        assert_eq!(m.len(), 7);
        let one_2 = m.remove().unwrap();
        assert_eq!(one_2, 12);
        assert_eq!(m.len(), 6);
        m.to_heap(1);

        let new_best_option = m.remove().unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(new_best_option % 10, 1);
    }

    #[test]
    fn test_bfs() {
        let mut m: MultiStageQueue<usize,i32> = MultiStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);

        m.to_heap_bfs();

        assert_eq!(m.remove().unwrap(), 1);
        assert_eq!(m.remove().unwrap(), 2);
        assert_eq!(m.remove().unwrap(), 3);
    }

    #[test]
    fn test_dfs() {
        let mut m: MultiStageQueue<usize,i32> = MultiStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);

        assert_eq!(m.remove().unwrap(), 3);
        assert_eq!(m.remove().unwrap(), 2);
        assert_eq!(m.remove().unwrap(), 1);
    }

}