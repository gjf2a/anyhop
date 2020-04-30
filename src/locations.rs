use immutable_map::TreeMap;
use crate::Atom;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct LocationGraph<L: Atom> {
    distances: TreeMap<L,TreeMap<L,usize>>
}

impl <L:Atom> LocationGraph<L> {
    pub fn new(distances: Vec<(L,L,usize)>) -> Self {
        let mut map_graph = LocationGraph {distances: TreeMap::new()};
        for distance in distances.iter() {
            map_graph.add(distance.0, distance.1, distance.2);
        }
        map_graph
    }

    pub fn get(&self, start: L, end: L) -> Option<usize> {
        self.distances.get(&start)
            .and_then(|map| map.get(&end))
            .map(|d| *d)
    }

    pub fn add(&mut self, m1: L, m2: L, distance: usize) {
        self.add_one_way(m1, m2, distance);
        self.add_one_way(m2, m1, distance);
    }

    fn add_one_way(&mut self, start: L, end: L, distance: usize) {
        let updated = self.distances.get(&start)
            .unwrap_or(&TreeMap::new())
            .insert(end, distance);
        self.distances = self.distances.insert(start, updated);
    }
}