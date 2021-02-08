use std::collections::HashMap;

pub struct Context<K, V> {
    inner: Vec<HashMap<K, V>>,
}

impl<K: Eq + std::hash::Hash, V> Context<K, V> {
    pub fn new() -> Self {
        Context { inner: vec![] }
    }

    pub fn push(&mut self) {
        self.inner.push(HashMap::new())

    }

    pub fn pop(&mut self) -> Option<HashMap<K, V>> {
        self.inner.pop()
    }

    pub fn insert(&mut self, key: K, value: V) -> () {
        self.inner.first_mut().unwrap().insert(key, value);
    }

    pub fn lookup(&mut self, key: &K) -> Option<&V> {
       for scope in (&self.inner).into_iter().rev() {
           match scope.get(key) {
               None => continue,
               Some(ty) => return Some(ty),
           }
       }

       None
    }
}
