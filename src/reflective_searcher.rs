// Implements a reflective searcher data structure.
//
// This data structure supports the algorithm in my research notebook.

use std::collections::{BTreeMap, VecDeque};

#[derive(Debug,Clone)]
pub struct TwoStageQueue<C:Ord,T> {
    backtrack_stack: Vec<T>,
    fifo: VecDeque<T>,
    prioritized: BTreeMap<C,Vec<T>>,
    size: usize
}

impl <C:Ord,T> TwoStageQueue<C,T> {
    pub fn new() -> Self {
        TwoStageQueue {backtrack_stack: Vec::new(), fifo: VecDeque::new(), prioritized: BTreeMap::new(), size: 0}
    }

    pub fn len(&self) -> usize { self.size }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, obj: T) {
        self.backtrack_stack.push(obj);
        self.size += 1;
    }

    pub fn remove(&mut self) -> Option<T> {
        let result = if self.backtrack_stack.is_empty() {
            if self.fifo.is_empty() {
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
                self.fifo.pop_front()
            }
        } else {
            self.backtrack_stack.pop()
        };
        if let Some(_) = result {
            self.size -= 1;
        }
        result
    }

    pub fn to_heap(&mut self, cost: C) {
        match self.prioritized.get_mut(&cost) {
            None => {self.prioritized.insert(cost, self.backtrack_stack.drain(..).collect());},
            Some(v) => self.backtrack_stack.drain(..).for_each(|t| v.push(t))
        }
    }

    pub fn to_bfs(&mut self) {
        let mut drained: Vec<T> = self.backtrack_stack.drain(..).collect();
        drained.drain(..).for_each(|item| self.fifo.push_back(item));
    }
}

#[cfg(test)]
mod tests {
    use crate::reflective_searcher::TwoStageQueue;

    #[test]
    fn test1() {
        let mut m = TwoStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);
        assert_eq!(m.len(), 3);
        let three = m.remove().unwrap();
        assert_eq!(three, 3);
        assert_eq!(m.len(), 2);
        m.to_heap(0);
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
        let mut m: TwoStageQueue<usize,i32> = TwoStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);

        m.to_bfs();

        assert_eq!(m.remove().unwrap(), 1);
        assert_eq!(m.remove().unwrap(), 2);
        assert_eq!(m.remove().unwrap(), 3);
    }

    #[test]
    fn test_dfs() {
        let mut m: TwoStageQueue<usize,i32> = TwoStageQueue::new();
        m.insert(1);
        m.insert(2);
        m.insert(3);

        assert_eq!(m.remove().unwrap(), 3);
        assert_eq!(m.remove().unwrap(), 2);
        assert_eq!(m.remove().unwrap(), 1);
    }

}