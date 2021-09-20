use egg::{define_language, rewrite, EGraph, Id, SimpleScheduler};
use itertools::Itertools;
use num::integer::Roots;
use rand::Rng;
use rand_pcg::Pcg64;
use ruler::{letter, map, self_product, CVec, SynthAnalysis, SynthLanguage, Synthesizer};
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hash::BuildHasherDefault;
use std::str::FromStr;

#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
pub enum Value {
    // starts with i
    Int(i32),
    // starts with [
    List(Vec<Value>),
    // starts with <
    Vec(Vec<Value>),
    // starts with b
    Bool(bool),
}

impl FromStr for Value {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, <Self as FromStr>::Err> {
        if s.starts_with("i") {
            let v: i32 = s[1..].parse().map_err(|_| "Bad integer.".to_string())?;
            Ok(Value::Int(v))
        } else {
            Err(format!("{} didn't match anything.", s))
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::List(l) => write!(f, "{:?}", l),
            Value::Vec(v) => write!(f, "<{:?}>", v),
        }
    }
}

fn split_into_halves(n: usize) -> (usize, usize) {
    if n % 2 == 0 {
        (n / 2, n / 2)
    } else {
        (n / 2, n / 2 + 1)
    }
}

impl Value {
    fn int1<F>(arg: &Self, f: F) -> Option<Value>
    where
        F: Fn(i32) -> Value,
    {
        if let Value::Int(val) = arg {
            Some(f(*val))
        } else {
            None
        }
    }

    fn int2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(i32, i32) -> Value,
    {
        if let (Value::Int(lv), Value::Int(rv)) = (lhs, rhs) {
            Some(f(*lv, *rv))
        } else {
            None
        }
    }

    fn bool2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(bool, bool) -> Value,
    {
        if let (Value::Bool(lv), Value::Bool(rv)) = (lhs, rhs) {
            Some(f(*lv, *rv))
        } else {
            None
        }
    }

    fn vec1<F>(val: &Self, f: F) -> Option<Value>
    where
        F: Fn(&[Value]) -> Value,
    {
        if let Value::Vec(v) = val {
            Some(f(v))
        } else {
            None
        }
    }

    fn vec2<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(&[Value], &[Value]) -> Option<Value>,
    {
        if let (Value::Vec(v1), Value::Vec(v2)) = (lhs, rhs) {
            if v1.len() == v2.len() {
                f(v1, v2)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn vec2_op<F>(lhs: &Self, rhs: &Self, f: F) -> Option<Value>
    where
        F: Fn(&Value, &Value) -> Option<Value>,
    {
        Self::vec2(lhs, rhs, |lhs, rhs| {
            lhs.iter()
                .zip(rhs)
                .map(|(l, r)| f(l, r))
                .collect::<Option<Vec<Value>>>()
                .map(|v| Value::Vec(v))
        })
    }

    fn sample_int(rng: &mut Pcg64, min: i32, max: i32, num_samples: usize) -> Vec<Value> {
        (0..num_samples)
            .map(|_| Value::Int(rng.gen_range(min, max)))
            .collect::<Vec<_>>()
    }

    fn sample_vec(
        rng: &mut Pcg64,
        min: i32,
        max: i32,
        list_size: usize,
        num_samples: usize,
    ) -> Vec<Value> {
        (0..num_samples)
            .map(|_| Value::Vec(Value::sample_int(rng, min, max, list_size)))
            .collect::<Vec<_>>()
    }
}

fn sgn(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x == 0 {
        0
    } else {
        -1
    }
}

define_language! {
    pub enum VecLang {
        // Id is a key to identify EClasses within an EGraph, represents
        // children nodes
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Minus([Id; 2]),
        "/" = Div([Id; 2]),

        "or" = Or([Id; 2]),
        "&&" = And([Id; 2]),
        "ite" = Ite([Id; 3]),
        "<" = Lt([Id; 2]),

        "sgn" = Sgn([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "neg" = Neg([Id; 1]),

        // Lists have a variable number of elements
        "List" = List(Box<[Id]>),

        // Vectors have width elements
        "Vec" = Vec(Box<[Id]>),

        // Vector with all literals
        // "LitVec" = LitVec(Box<[Id]>),

        "Get" = Get([Id; 2]),

        // Used for partitioning and recombining lists
        "Concat" = Concat([Id; 2]),

        // Vector operations that take 2 vectors of inputs
        "VecAdd" = VecAdd([Id; 2]),
        "VecMinus" = VecMinus([Id; 2]),
        "VecMul" = VecMul([Id; 2]),
        "VecDiv" = VecDiv([Id; 2]),
        // "VecMulSgn" = VecMulSgn([Id; 2]),

        // Vector operations that take 1 vector of inputs
        "VecNeg" = VecNeg([Id; 1]),
        "VecSqrt" = VecSqrt([Id; 1]),
        "VecSgn" = VecSgn([Id; 1]),

        // MAC takes 3 lists: acc, v1, v2
        // "VecMAC" = VecMAC([Id; 3]),

        Const(Value),

        // language items are parsed in order, and we want symbol to
        // be a fallback, so we put it last.
        // `Symbol` is an egg-provided interned string type
        Symbol(egg::Symbol),
    }
}

impl SynthLanguage for VecLang {
    type Constant = Value;

    fn to_var(&self) -> Option<egg::Symbol> {
        if let VecLang::Symbol(sym) = self {
            Some(*sym)
        } else {
            None
        }
    }

    fn mk_var(sym: egg::Symbol) -> Self {
        VecLang::Symbol(sym)
    }

    fn to_constant(&self) -> Option<&Self::Constant> {
        if let VecLang::Const(n) = self {
            Some(n)
        } else {
            None
        }
    }

    fn mk_constant(c: <Self as SynthLanguage>::Constant) -> Self {
        VecLang::Const(c)
    }

    fn eval<'a, F>(&'a self, cvec_len: usize, mut get: F) -> CVec<Self>
    where
        F: FnMut(&'a Id) -> &'a CVec<Self>,
    {
        match self {
            VecLang::Const(i) => vec![Some(i.clone()); cvec_len],
            VecLang::Add([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l + r))),
            VecLang::Mul([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l * r))),
            VecLang::Minus([l, r]) => {
                map!(get, l, r => Value::int2(l, r, |l, r| Value::Int(l - r)))
            }
            VecLang::Div([l, r]) => get(l)
                .iter()
                .zip(get(r).iter())
                .map(|tup| match tup {
                    (Some(Value::Int(a)), Some(Value::Int(b))) => {
                        if *b != 0 {
                            Some(Value::Int(a / b))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            VecLang::Or([l, r]) => {
                map!(get, l, r => Value::bool2(l, r, |l, r| Value::Bool(l || r)))
            }
            VecLang::And([l, r]) => {
                map!(get, l, r => Value::bool2(l, r, |l, r| Value::Bool(l && r)))
            }
            VecLang::Ite([_b, _t, _f]) => todo!(),
            VecLang::Lt([l, r]) => map!(get, l, r => Value::int2(l, r, |l, r| Value::Bool(l < r))),
            VecLang::Sgn([x]) => {
                map!(get, x => Value::int1(x, |x| Value::Int(sgn(x))))
            }
            VecLang::Sqrt([x]) => get(x)
                .iter()
                .map(|a| match a {
                    Some(Value::Int(a)) => {
                        if *a >= 0 {
                            Some(Value::Int(a.sqrt()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            VecLang::Neg([x]) => map!(get, x => Value::int1(x, |x| Value::Int(-x))),
            VecLang::List(l) => l
                .iter()
                .fold(vec![Some(vec![]); cvec_len], |mut acc, item| {
                    acc.iter_mut().zip(get(item)).for_each(|(mut v, i)| {
                        if let (Some(v), Some(i)) = (&mut v, i) {
                            v.push(i.clone());
                        } else {
                            *v = None;
                        }
                    });
                    acc
                })
                .into_iter()
                .map(|acc| {
                    if let Some(x) = acc {
                        Some(Value::List(x))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            VecLang::Vec(l) => l
                .iter()
                .fold(vec![Some(vec![]); cvec_len], |mut acc, item| {
                    acc.iter_mut().zip(get(item)).for_each(|(mut v, i)| {
                        if let (Some(v), Some(i)) = (&mut v, i) {
                            v.push(i.clone());
                        } else {
                            *v = None;
                        }
                    });
                    acc
                })
                .into_iter()
                .map(|acc| {
                    if let Some(x) = acc {
                        Some(Value::Vec(x))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            // VecLang::LitVec(x) => todo!(),
            VecLang::Get([l, i]) => map!(get, l, i => {
                if let (Value::Vec(v), Value::Int(idx)) = (l, i) {
            // get index and clone the inner Value if there is one
                    v.get(*idx as usize).map(|inner| inner.clone())
                } else {
            None
            }
                }),
            VecLang::Concat([l, r]) => {
                map!(get, l, r => Value::vec2(l, r, |l, r| {
                Some(Value::List(l.iter().chain(r).cloned().collect::<Vec<_>>()))
                    }))
            }
            VecLang::VecAdd([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l + r))))
            }
            VecLang::VecMinus([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l - r))))
            }
            VecLang::VecMul([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| Value::int2(l, r, |l, r| Value::Int(l * r))))
            }
            VecLang::VecDiv([l, r]) => {
                map!(get, l, r => Value::vec2_op(l, r, |l, r| {
                match (l, r) {
                    (Value::Int(a), Value::Int(b)) => {
                        if *b != 0 {
                            Some(Value::Int(a / b))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
                }))
            }
            VecLang::VecNeg([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(-a),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            VecLang::VecSqrt([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(a.sqrt()),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            VecLang::VecSgn([l]) => {
                map!(get, l => Value::vec1(l, |l| {
                    Value::Vec(l.iter().map(|tup| match tup {
                    Value::Int(a) => Value::Int(sgn(*a)),
                        _ => panic!("Ill-formed")
                    }).collect::<Vec<_>>())
                }))
            }
            // VecLang::VecMAC([a, b, c]) => todo!(),
            VecLang::Symbol(_) => vec![],
        }
    }

    fn init_synth(synth: &mut Synthesizer<Self>) {
        let consts: [Value; 2] = [
            // Value::List(vec![]),
            // Value::Vec(vec![Value::Int(0); synth.params.vector_size]),
            // Value::Vec(vec![Value::Int(1); synth.params.vector_size]),
            // Value::Vec(vec![]),
            // Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            // Value::Bool(true),
            // Value::Bool(false),
        ];

        // initial set of rewrite rules
        // let iso_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(VecAdd (Vec ?a ?b) (Vec ?c ?d))".parse().unwrap(),
        //     &"(Vec (+ ?a ?c) (+ ?b ?d))".parse().unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("iso_add".into(), iso_add);

        // let iso_add: ruler::Equality<VecLang> = ruler::Equality::new(
        //     &"(VecAdd ?a ?b)".parse().unwrap(),
        //     &"(VecAdd (Vec (Get ?a i0) (Get ?a i1)) (Vec (Get ?b i0) (Get ?b i1)))"
        //         .parse()
        //         .unwrap(),
        // )
        // .unwrap();
        // synth.equalities.insert("iso_add".into(), iso_add);

        let comm_add: ruler::Equality<VecLang> =
            ruler::Equality::new(&"(+ ?a ?b)".parse().unwrap(), &"(+ ?b ?a)".parse().unwrap())
                .unwrap();
        synth.equalities.insert("comm_add".into(), comm_add);

        let consts_cross = self_product(
            &consts.iter().map(|x| Some(x.clone())).collect::<Vec<_>>(),
            synth.params.variables,
        );

        let size = consts_cross[0].len();

        // new egraph
        let mut egraph = EGraph::new(SynthAnalysis { cvec_len: size });

        let a = egraph.add_expr(&"(VecAdd (Vec ?a ?b) (Vec ?c ?d))".parse().unwrap());
        let b = egraph.add_expr(&"(VecAdd (Vec ?a ?d) (Vec ?c ?b))".parse().unwrap());
        egraph.add_expr(&"(Vec (+ ?a ?c) (+ ?b ?d))".parse().unwrap());
        egraph.add_expr(&"(Vec (+ ?a ?c) (+ ?d ?b))".parse().unwrap());
        egraph.add_expr(&"(+ ?a ?b)".parse().unwrap());
        egraph.add_expr(&"(+ ?b ?a)".parse().unwrap());

        let rewrites = vec![
            rewrite!("add-comm"; "(+ ?a ?b)" <=> "(+ ?b ?a)"),
            rewrite!("add-vec-iso";
		     "(VecAdd (Vec ?a ?b) (Vec ?c ?d))" <=> "(Vec (+ ?a ?c) (+ ?b ?d))"),
        ]
        .concat();

        log::info!("before {:?} ?= {:?}", egraph.find(a), egraph.find(b));

        let runner = egg::Runner::default()
            .with_iter_limit(10)
            .with_node_limit(10_000)
            .with_egraph(egraph)
            .with_scheduler(SimpleScheduler)
            .run(&rewrites);

        runner.egraph.dot().to_png("play.png").unwrap();

        log::info!(
            "after {:?} ?= {:?}",
            runner.egraph.find(a),
            runner.egraph.find(b)
        );

        panic!();

        // add constants
        for v in consts.iter() {
            egraph.add(VecLang::Const(v.clone()));
        }

        // add variables
        for i in 0..synth.params.variables {
            let var = egg::Symbol::from(letter(i));
            let id = egraph.add(VecLang::Symbol(var));

            // add gets for each variable
            // for l in 0..synth.params.vector_size {
            //     let idx_id = egraph.add(VecLang::Const(Value::Int(l as i32)));
            //     log::info!("(GET {} {}) {:?}", var, l, VecLang::Get([id, idx_id]));
            //     egraph.add(VecLang::Get([id, idx_id]));
            // }

            // make the cvec use real data
            let mut cvec = vec![];

            let (n_ints, n_vecs) = split_into_halves(size);

            cvec.extend(
                Value::sample_int(&mut synth.rng, -100, 100, n_ints)
                    .into_iter()
                    .map(Some),
            );

            cvec.extend(
                Value::sample_vec(&mut synth.rng, -100, 100, synth.params.vector_size, n_vecs)
                    .into_iter()
                    .map(Some),
            );

            egraph[id].data.cvec = cvec;
        }

        // set egraph to the one we just constructed
        synth.egraph = egraph;
    }

    fn make_layer<'a>(
        ids: Vec<Id>,
        synth: &'a Synthesizer<Self>,
        _iter: usize,
    ) -> Box<dyn Iterator<Item = Self> + 'a> {
        let binops = (0..2)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
            .map(|ids| [ids[0], ids[1]])
            .map(|x| {
                vec![
                    VecLang::Add(x),
                    // VecLang::Minus(x),
                    // VecLang::Mul(x),
                    // VecLang::Concat(x),
                    VecLang::VecAdd(x),
                    // VecLang::VecMinus(x),
                ]
            })
            .flatten();

        let vec = (0..synth.params.vector_size)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| !ids.iter().all(|x| synth.egraph[*x].data.exact))
            .map(|x| vec![VecLang::Vec(x.into_boxed_slice())])
            .flatten();

        Box::new(binops.chain(vec))
    }

    fn is_valid(
        synth: &mut Synthesizer<Self>,
        lhs: &egg::Pattern<Self>,
        rhs: &egg::Pattern<Self>,
    ) -> bool {
        // use fuzzing to determine equality

        let n = synth.params.num_fuzz;
        let mut env = HashMap::default();

        for var in lhs.vars() {
            env.insert(var, vec![]);
        }

        for var in rhs.vars() {
            env.insert(var, vec![]);
        }

        let (n_ints, n_vecs) = split_into_halves(n);

        for cvec in env.values_mut() {
            cvec.reserve(n);
            cvec.extend(
                Value::sample_int(&mut synth.rng, -100, 100, n_ints)
                    .into_iter()
                    .map(Some),
            );

            cvec.extend(
                Value::sample_vec(&mut synth.rng, -100, 100, synth.params.vector_size, n_vecs)
                    .into_iter()
                    .map(Some),
            );
        }

        // let left: Pattern<VecLang> = "(Vec (+ ?a ?b) (+ ?c ?d))".parse().unwrap();
        // let right: Pattern<VecLang> = "(VecAdd (Vec ?a ?c) (Vec ?b ?d))".parse().unwrap();
        // debug(
        //     "(VecAdd (Vec ?a ?b) (Vec ?b ?a))",
        //     "(Vec (+ ?a ?b) (+ ?a ?b))",
        //     n,
        //     &env,
        // );

        // let test_env: HashMap<egg::Var, Vec<Option<Value>>, BuildHasherDefault<_>> = vec![
        //     ("?a".parse().unwrap(), vec![Some(Value::Int(1))]),
        //     ("?b".parse().unwrap(), vec![Some(Value::Int(2))]),
        //     ("?c".parse().unwrap(), vec![Some(Value::Int(3))]),
        //     ("?d".parse().unwrap(), vec![Some(Value::Int(4))]),
        // ]
        // .into_iter()
        // .collect();
        // log::info!("env: {:?}", test_env);
        // log::info!("lhs: {}\nrhs: {}", left, right);
        // let lres = Self::eval_pattern(&left, &test_env, n);
        // let rres = Self::eval_pattern(&right, &test_env, n);
        // log::info!("{:?} ?= ({}) {:?}", lres, lres == rres, rres);
        // panic!("stop");

        let lvec = Self::eval_pattern(lhs, &env, n);
        let rvec = Self::eval_pattern(rhs, &env, n);

        if lvec != rvec {
            log::debug!("  env: {:?}", env);
            log::debug!("  lhs: {}, rhs: {}", lhs, rhs);
            log::debug!("  lvec: {:?}, rvec: {:?}", lvec, rvec);
        }

        // only compare values where both sides are defined
        if lvec.iter().all(|x| x.is_none()) && rvec.iter().all(|x| x.is_none()) {
            false
        } else {
            lvec.iter().zip(rvec.iter()).all(|tup| match tup {
                (Some(l), Some(r)) => l == r,
                _ => true,
            })
        }
    }
}

fn debug(
    left: &str,
    right: &str,
    n: usize,
    env: &HashMap<egg::Var, Vec<Option<Value>>, BuildHasherDefault<FxHasher>>,
) {
    let pleft: egg::Pattern<VecLang> = left.parse().unwrap();
    let pright: egg::Pattern<VecLang> = right.parse().unwrap();
    let lres = VecLang::eval_pattern(&pleft, env, n);
    let rres = VecLang::eval_pattern(&pright, env, n);
    log::info!(
        "TEST:\n  {:?}\n    ?= ({})\n  {:?}",
        lres,
        lres == rres,
        rres
    );
    panic!();
}

fn main() {
    VecLang::main()
}

// 132289, 15053463213406696608
