use egg::{define_language, EGraph, Id};
use itertools::Itertools;
use num::integer::Roots;
use rand::Rng;
use rand_pcg::Pcg64;
use ruler::{letter, map, self_product, CVec, SynthAnalysis, SynthLanguage, Synthesizer};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
pub enum Value {
    Int(i32),
    List(Vec<Value>),
    Vec(Vec<Value>),
    Bool(bool),
}

impl FromStr for Value {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, <Self as FromStr>::Err> {
        Err(format!("{} didn't match anything.", s))
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
        Const(Value),

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
        let consts: [Value; 3] = [
            // Value::List(vec![]),
            // Value::Vec(vec![]),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            // Value::Bool(true),
            // Value::Bool(false),
        ];

        let consts_cross = self_product(
            &consts.iter().map(|x| Some(x.clone())).collect::<Vec<_>>(),
            synth.params.variables,
        );

        let size = consts_cross[0].len();

        // new egraph
        let mut egraph = EGraph::new(SynthAnalysis { cvec_len: size });

        // add constants
        for v in consts.iter() {
            egraph.add(VecLang::Const(v.clone()));
        }

        // add variables
        for i in 0..synth.params.variables {
            let var = egg::Symbol::from(letter(i));
            let id = egraph.add(VecLang::Symbol(var));

            // make the cvec use real data
            let mut cvec = vec![];

            let (n_ints, n_vecs) = split_into_halves(size);

            cvec.extend(
                Value::sample_int(&mut synth.rng, -100, 100, n_ints)
                    .into_iter()
                    .map(Some),
            );

            cvec.extend(
                Value::sample_vec(&mut synth.rng, -100, 100, synth.params.num_fuzz, n_vecs)
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
        egraph: &'a EGraph<Self, SynthAnalysis>,
        _iter: usize,
    ) -> Box<dyn Iterator<Item = Self> + 'a> {
        let two = (0..2)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| !ids.iter().all(|x| egraph[*x].data.exact))
            .map(|ids| [ids[0], ids[1]])
            .map(|x| {
                vec![
                    VecLang::Add(x),
                    VecLang::Minus(x),
                    VecLang::Mul(x),
                    VecLang::Concat(x),
                    VecLang::VecAdd(x),
                    VecLang::VecMinus(x),
                    // VecLang::Vec(Box::new(x)),
                ]
            })
            .flatten();

        let four = (0..4)
            .map(|_| ids.clone())
            .multi_cartesian_product()
            .filter(move |ids| ids.iter().all(|x| !egraph[*x].data.exact))
            .map(|ids| [ids[0], ids[1], ids[2], ids[3]])
            .map(|x| vec![VecLang::Vec(Box::new(x))])
            .flatten();

        Box::new(two.chain(four))

        // let mut to_add = vec![];
        // log::info!("#ids: {}", synth.ids().count());
        // for i in synth.ids() {
        //     // one operand operators
        //     if !synth.egraph[i].data.exact {
        //         // to_add.push(VecLang::Sgn([i]));
        //         // to_add.push(VecLang::Sqrt([i]));
        //         // to_add.push(VecLang::Neg([i]));

        //         // size 1 lists
        //         // to_add.push(VecLang::List(Box::new([i])));
        //         // to_add.push(VecLang::Vec(Box::new([i])));
        //     }

        //     for j in synth.ids() {
        //         // two operand operators
        //         if !(synth.egraph[i].data.exact && synth.egraph[j].data.exact) {
        //             to_add.push(VecLang::Add([i, j]));
        //             to_add.push(VecLang::Mul([i, j]));
        //             to_add.push(VecLang::Minus([i, j]));
        //             // to_add.push(VecLang::Div([i, j]));

        //             // to_add.push(VecLang::Or([i, j]));
        //             // to_add.push(VecLang::And([i, j]));
        //             // to_add.push(VecLang::Lt([i, j]));

        //             // to_add.push(VecLang::Get([i, j]));
        //             to_add.push(VecLang::Concat([i, j]));
        //             to_add.push(VecLang::VecAdd([i, j]));
        //             to_add.push(VecLang::VecMinus([i, j]));
        //             // to_add.push(VecLang::VecMul([i, j]));
        //             // to_add.push(VecLang::VecDiv([i, j]));

        //             // size two lists
        //             // to_add.push(VecLang::List(Box::new([i, j])));
        //             to_add.push(VecLang::Vec(Box::new([i, j])));
        //         }

        //         for k in synth.ids() {
        //             for l in synth.ids() {
        //                 if !(synth.egraph[i].data.exact
        //                     && synth.egraph[j].data.exact
        //                     && synth.egraph[k].data.exact
        //                     && synth.egraph[l].data.exact)
        //                 {
        //                     to_add.push(VecLang::Vec(Box::new([i, j, k, l])));
        //                 }
        //             }
        //         }
        //         // for k in synth.ids() {
        //         //     // size 3 operators
        //         //     if !(synth.egraph[i].data.exact
        //         //         && synth.egraph[j].data.exact
        //         //         && synth.egraph[k].data.exact)
        //         //     {
        //         //         to_add.push(VecLang::List(Box::new([i, j, k])));
        //         //         to_add.push(VecLang::Vec(Box::new([i, j, k])));
        //         //     }
        //         // }
        //     }
        // }

        // log::info!("Made a layer of {} enodes", to_add.len());
        // to_add
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
                Value::sample_vec(&mut synth.rng, -100, 100, 4, n_vecs)
                    .into_iter()
                    .map(Some),
            );
        }

        // let left: Pattern<VecLang> = "(Vec (+ ?a ?b) (+ ?c ?d))".parse().unwrap();
        // let right: Pattern<VecLang> = "(VecAdd (Vec ?a ?c) (Vec ?b ?d))".parse().unwrap();

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

        lvec == rvec
    }
}

fn main() {
    VecLang::main()
}

// 132289, 15053463213406696608
