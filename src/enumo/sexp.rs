use std::{
    collections::{LinkedList, VecDeque},
    str::FromStr,
};

use super::*;

/// S-expression
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Sexp {
    Atom(String),
    List(Vec<Self>),
}

impl FromStr for Sexp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use symbolic_expressions::parser::parse_str;
        let sexp = parse_str(s).unwrap();
        Ok(Self::from_symbolic_expr(sexp))
    }
}

impl std::fmt::Display for Sexp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sexp::Atom(x) => write!(f, "{}", x),
            Sexp::List(l) => {
                write!(f, "(").expect("not written");
                for x in l {
                    write!(f, "{} ", x).expect("not written");
                }
                write!(f, ")").expect("not written");
                Ok(())
            }
        }
    }
}

static SSI_COUNT: std::sync::RwLock<usize> = std::sync::RwLock::new(0);

#[derive(Clone)]
pub struct SexpSubstIter<I, F>
where
    I: Iterator<Item = Sexp>,
    F: Fn() -> I,
{
    needle: String,
    spawn_iterator: F,
    stack: VecDeque<(Sexp, I)>,
    level: usize,
}

impl<I, F> SexpSubstIter<I, F>
where
    I: Iterator<Item = Sexp>,
    F: Fn() -> I,
{
    pub fn new<S: ToString>(inital_sexp: Sexp, needle: S, spawn_iterator: F) -> Self {
        let initial_iter = spawn_iterator();
        let level = SSI_COUNT.read().unwrap().clone();
        let mut ssi_count_ref = SSI_COUNT.write().unwrap();
        *ssi_count_ref += 1;
        // println!("starting ssi for {inital_sexp} \\ {}", needle.to_string());
        SexpSubstIter {
            needle: needle.to_string(),
            spawn_iterator,
            stack: VecDeque::from([(inital_sexp, initial_iter)]),
            level,
        }
    }
}

impl<I, F> Iterator for SexpSubstIter<I, F>
where
    I: Iterator<Item = Sexp>,
    F: Fn() -> I,
{
    type Item = Sexp;

    fn next(&mut self) -> Option<Self::Item> {
        // println!("/--{}-------- {}", self.level, self.needle);
        // for (template, _) in &self.stack {
        //     println!("* {template}");
        // }
        // println!("\\-----------");

        self.stack
            .pop_front()
            .and_then(|(parent_sexp, mut parent_iter)| {
                let mut parent_clone = parent_sexp.clone();
                let mut needle_refs = parent_clone.find_all(&self.needle);

                // if there are no instances of the needle, we can return immediately
                if needle_refs.is_empty() {
                    // println!("semi-yield {parent_clone}");
                    Some(parent_clone)
                } else {
                    // now we know that there is at least one instance of the needle

                    // if there is something left in the parent_iter
                    if let Some(next_item) = parent_iter.next() {
                        self.stack.push_front((parent_sexp, parent_iter));
                        if let Some(ptr) = needle_refs.pop_front() {
                            *ptr = next_item;
                            if !needle_refs.is_empty() {
                                let child_iter = (self.spawn_iterator)();
                                self.stack.push_front((parent_clone, child_iter));
                                self.next()
                            } else {
                                // println!("semi-yield {parent_clone}");
                                Some(parent_clone)
                            }
                        } else {
                            None
                        }
                    } else {
                        // skip this item
                        self.next()
                    }
                }
                // // if there is juice left in the iterator
                // if let Some(next_item) = parent_iter.next() {
                //     // try to go deeper one layer by replacing the first instance of the
                //     // needle with the item we got from the iterator
                //     if let Some((child_sexp, more)) =
                //         parent_sexp.replace_first(&self.needle, &next_item)
                //     {
                //         // there might be more juice in the parent_iter,
                //         // so push it back on the stack so that we try
                //         // to process it again
                //         self.stack.push_front((parent_sexp, parent_iter));

                //         // next we want to spawn a new iterator representing one layer
                //         // deeper in the search. we want to make sure that this item
                //         // is the next item processed on the stack so that we perform
                //         // a depth-first traversal of the tree.
                //         if more {
                //             let child_iter = (self.spawn_iterator)();
                //             self.stack.push_front((child_sexp, child_iter));
                //             self.next()
                //         } else {
                //             // self.stack.push_front((child_sexp, None));
                //             println!("yield {} child", self.level);
                //             Some(child_sexp)
                //         }
                //     } else {
                //         println!("yield {} parent {parent_sexp}", self.level);
                //         // otherwise (no needle), we are at a leaf and all instances
                //         // of the needle are fully instantiated. we can yield this
                //         // item from the iterator
                //         Some(parent_sexp)
                //     }
                // } else {
                //     // we are done with this layer of the tree. continue processing
                //     // the next item on the stack
                //     self.next()
                // }
            })
    }
}

impl Sexp {
    fn from_symbolic_expr(sexp: symbolic_expressions::Sexp) -> Self {
        match sexp {
            symbolic_expressions::Sexp::String(s) => Self::Atom(s),
            symbolic_expressions::Sexp::List(ss) => Self::List(
                ss.iter()
                    .map(|s| Sexp::from_symbolic_expr(s.clone()))
                    .collect(),
            ),
            symbolic_expressions::Sexp::Empty => Self::List(vec![]),
        }
    }

    fn mk_canon(
        &self,
        symbols: &[String],
        mut idx: usize,
        mut subst: HashMap<String, String>,
    ) -> (HashMap<String, String>, usize) {
        match self {
            Sexp::Atom(x) => {
                if symbols.contains(x) && !subst.contains_key(x) {
                    subst.insert(x.into(), symbols[idx].clone());
                    idx += 1;
                }
                (subst, idx)
            }
            Sexp::List(exps) => exps.iter().fold((subst, idx), |(acc, idx), item| {
                item.mk_canon(symbols, idx, acc)
            }),
        }
    }

    fn apply_subst(&self, subst: &HashMap<String, String>) -> Self {
        match self {
            Sexp::Atom(s) => {
                if let Some(v) = subst.get(s) {
                    Sexp::Atom(v.into())
                } else {
                    Sexp::Atom(s.into())
                }
            }
            Sexp::List(exps) => Sexp::List(exps.iter().map(|s| s.apply_subst(subst)).collect()),
        }
    }

    pub(crate) fn canon(&self, symbols: &[String]) -> Self {
        let (subst, _) = self.mk_canon(symbols, 0, Default::default());
        self.apply_subst(&subst)
    }

    pub(crate) fn plug(&self, name: &str, pegs: &[Self]) -> Vec<Sexp> {
        use itertools::Itertools;
        match self {
            Sexp::Atom(s) if s == name => pegs.to_vec(),
            Sexp::Atom(_) => vec![self.clone()],
            Sexp::List(sexps) => sexps
                .iter()
                .map(|x| x.plug(name, pegs))
                .multi_cartesian_product()
                .map(Sexp::List)
                .collect(),
        }
    }

    pub(crate) fn measure(&self, metric: Metric) -> usize {
        match self {
            Sexp::Atom(_) => match metric {
                Metric::Lists => 0,
                Metric::Atoms | Metric::Depth => 1,
            },
            Sexp::List(s) => match metric {
                Metric::Atoms => s.iter().map(|x| x.measure(metric)).sum::<usize>(),
                Metric::Lists => s.iter().map(|x| x.measure(metric)).sum::<usize>() + 1,
                Metric::Depth => s.iter().map(|x| x.measure(metric)).max().unwrap() + 1,
            },
        }
    }

    fn first(&mut self, needle: &str) -> Option<&mut Self> {
        match self {
            Sexp::Atom(a) if a == needle => Some(self),
            Sexp::Atom(_) => None,
            Sexp::List(list) => list.into_iter().find_map(|s| s.first(needle)),
        }
    }

    pub(crate) fn find_all(&mut self, needle: &str) -> LinkedList<&mut Self> {
        match self {
            Sexp::Atom(a) if a == needle => LinkedList::from([self]),
            Sexp::Atom(_) => LinkedList::new(),
            Sexp::List(list) => list.into_iter().fold(LinkedList::new(), |mut acc, el| {
                acc.append(&mut el.find_all(needle));
                acc
            }),
        }
    }

    pub(crate) fn replace_first(&self, needle: &str, new: &Sexp) -> Option<(Self, bool)> {
        let mut copy = self.clone();

        let mut all_refs = copy.find_all(needle);
        if let Some(first) = all_refs.pop_front() {
            *first = new.clone();
            let empty = all_refs.is_empty();
            Some((copy, !empty))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_str() {
        assert_eq!("a".parse::<Sexp>().unwrap(), Sexp::Atom("a".into()));
        assert_eq!(
            "(+ (- 1 2) 0)".parse::<Sexp>().unwrap(),
            Sexp::List(vec![
                Sexp::Atom("+".into()),
                Sexp::List(vec![
                    Sexp::Atom("-".into()),
                    Sexp::Atom("1".into()),
                    Sexp::Atom("2".into()),
                ]),
                Sexp::Atom("0".into()),
            ])
        )
    }

    #[test]
    fn measure_atoms() {
        let exprs = vec![
            ("a", 1),
            ("(a b)", 2),
            ("(a b c)", 3),
            ("(a (b c))", 3),
            ("(a b (c d))", 4),
            ("(a (b (c d)))", 4),
            ("(a (b c) (d e))", 5),
        ];
        for (expr, size) in exprs {
            assert_eq!(expr.parse::<Sexp>().unwrap().measure(Metric::Atoms), size);
        }
    }

    #[test]
    fn measure_lists() {
        let exprs = vec![
            ("a", 0),
            ("(a b)", 1),
            ("(a b c)", 1),
            ("(a (b c))", 2),
            ("(a b (c d))", 2),
            ("(a (b (c d)))", 3),
            ("(a (b c) (d e))", 3),
        ];
        for (expr, size) in exprs {
            assert_eq!(expr.parse::<Sexp>().unwrap().measure(Metric::Lists), size);
        }
    }

    #[test]
    fn measure_depth() {
        let exprs = vec![
            ("a", 1),
            ("(a b)", 2),
            ("(a b c)", 2),
            ("(a (b c))", 3),
            ("(a b (c d))", 3),
            ("(a (b (c d)))", 4),
            ("(a (b c) (d e))", 3),
        ];
        for (expr, size) in exprs {
            assert_eq!(expr.parse::<Sexp>().unwrap().measure(Metric::Depth), size);
        }
    }

    #[test]
    fn plug() {
        let x = "x".parse::<Sexp>().unwrap();
        let pegs = Workload::new(["1", "2", "3"]).force();
        let expected = vec![x.clone()];
        let actual = x.plug("a", &pegs);
        assert_eq!(actual, expected);

        let expected = pegs.clone();
        let actual = x.plug("x", &pegs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn plug_cross_product() {
        let term = "(x x)";
        let pegs = Workload::new(["1", "2", "3"]).force();
        let expected = Workload::new([
            "(1 1)", "(1 2)", "(1 3)", "(2 1)", "(2 2)", "(2 3)", "(3 1)", "(3 2)", "(3 3)",
        ])
        .force();
        let actual = term.parse::<Sexp>().unwrap().plug("x", &pegs);
        assert_eq!(actual, expected);
    }

    #[test]
    fn multi_plug() {
        let wkld = Workload::new(["(a b)", "(a)", "(b)"]);
        let a_s = Workload::new(["1", "2", "3"]);
        let b_s = Workload::new(["x", "y"]);
        let actual = wkld.plug("a", &a_s).plug("b", &b_s).force();
        let expected = Workload::new([
            "(1 x)", "(1 y)", "(2 x)", "(2 y)", "(3 x)", "(3 y)", "(1)", "(2)", "(3)", "(x)", "(y)",
        ])
        .force();
        assert_eq!(actual, expected)
    }

    #[test]
    fn canon() {
        let inputs = Workload::new([
            "(+ (/ c b) a)",
            "(+ (- c c) (/ a a))",
            "a",
            "b",
            "x",
            "(+ a a)",
            "(+ b b)",
            "(+ a b)",
            "(+ b a)",
            "(+ a x)",
            "(+ x a)",
            "(+ b x)",
            "(+ x b)",
            "(+ a (+ b c))",
            "(+ a (+ c b))",
        ])
        .force();
        let expecteds = Workload::new([
            "(+ (/ a b) c)",
            "(+ (- a a) (/ b b))",
            "a",
            "a",
            "x",
            "(+ a a)",
            "(+ a a)",
            "(+ a b)",
            "(+ a b)",
            "(+ a x)",
            "(+ x a)",
            "(+ a x)",
            "(+ x a)",
            "(+ a (+ b c))",
            "(+ a (+ b c))",
        ])
        .force();
        for (test, expected) in inputs.iter().zip(expecteds.iter()) {
            assert_eq!(
                &test.canon(vec!["a".into(), "b".into(), "c".into()].as_ref()),
                expected
            );
        }
    }
}
