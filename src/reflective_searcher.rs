// Implements a reflective searcher data structure.
//
// This data structure supports the algorithm in my research notebook.

// I need to start by creating an API and unit tests.

use std::collections::BTreeMap;
use crate::Cost;

struct MultiStageQueue<C:Cost,T> {
    backtrack_stack: Vec<T>,
    holding_area_1: Vec<T>,
    holding_area_2: Vec<T>,
    prioritized: BTreeMap<C,Vec<T>>
}

impl <C:Cost,T> MultiStageQueue<C,T> {
    pub fn new() -> Self {
        MultiStageQueue {backtrack_stack: Vec::new(), holding_area_1: Vec::new(),
        holding_area_2: Vec::new(), prioritized: BTreeMap::new()}
    }

    pub fn insert(&mut self, obj: T) {
        self.backtrack_stack.push(obj);
    }

    pub fn remove(&mut self) -> Option<T> {
        if self.backtrack_stack.is_empty() {
            if self.holding_area_1.is_empty() {
                if self.holding_area_2.is_empty() {
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
                    self.holding_area_2.pop()
                }
            } else {
                self.holding_area_1.pop()
            }
        } else {
            self.backtrack_stack.pop()
        }
    }

    pub fn to_holding_1(&mut self) {
        let mut moving: Vec<T> = self.backtrack_stack.drain(..).collect();
        self.holding_area_1.append(&mut moving);
    }

    pub fn to_holding_2(&mut self) {
        let mut moving: Vec<T> = self.backtrack_stack.drain(..).collect();
        self.holding_area_2.append(&mut moving);
    }

    pub fn holding_1_to_heap(&mut self, cost: C) {
        match self.prioritized.get_mut(&cost) {
            None => {self.prioritized.insert(cost, self.holding_area_1.drain(..).collect());},
            Some(v) => self.holding_area_1.drain(..).for_each(|t| v.push(t))
        }
    }
}