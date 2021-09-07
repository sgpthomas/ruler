use egg::{define_language, Id};
use num::integer::Roots;
use ruler::{map, CVec, SynthLanguage, Synthesizer};
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Value {
    Int(i32),
    List(Vec<Value>),
    Bool(bool),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::List(l) => write!(f, "{:?}", l),
        }
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

    fn sqrt(&self) -> Option<Value> {
        if let Value::Int(val) = self {
            Some(Value::Int(val.sqrt()))
        } else {
            None
        }
    }
}

define_language! {
    pub enum VecLang {
        Num(i32),

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
        "LitVec" = LitVec(Box<[Id]>),

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
        "VecMAC" = VecMAC([Id; 3]),

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
        if let VecLang::Num(n) = self {
            Some(&Value::Int(*n))
        } else {
            None
        }
    }

    fn mk_constant(c: <Self as SynthLanguage>::Constant) -> Self {
        match c {
            Value::Int(i) => VecLang::Num(i),
            Value::Bool(_) | Value::List(_) => panic!("oh woe become me!"),
        }
    }

    fn init_synth(synth: &mut Synthesizer<Self>) {
        todo!("init_synth")
    }

    fn eval<'a, F>(&'a self, cvec_len: usize, mut v: F) -> CVec<Self>
    where
        F: FnMut(&'a Id) -> &'a CVec<Self>,
    {
        match self {
            VecLang::Num(i) => vec![Some(Value::Int(*i)); cvec_len],
            VecLang::Add([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l + r))),
            VecLang::Mul([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l * r))),
            VecLang::Minus([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l - r))),
            VecLang::Div([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l / r))),
            VecLang::Or([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l | r))),
            VecLang::And([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Int(l & r))),
            // VecLang::Ite([a, b, c]) => todo!(),
            VecLang::Lt([l, r]) => map!(v, l, r => Value::int2(l, r, |l, r| Value::Bool(l < r))),
            // VecLang::Sgn([x]) => todo!(),
            VecLang::Sqrt([x]) => map!(v, x => Value::int1(x, |x| Value::Int(x.sqrt()))),
            VecLang::Neg([x]) => map!(v, x => Value::int1(x, |x| Value::Int(-x))),
            VecLang::List(l) => {
                vec![Some(Value::List())]
            }
            VecLang::Vec(l) => todo!(),
            VecLang::LitVec(x) => todo!(),

            VecLang::Get([l, r]) => todo!(),
            VecLang::Concat([l, r]) => todo!(),
            VecLang::VecAdd([l, r]) => todo!(),
            VecLang::VecMinus([l, r]) => todo!(),
            VecLang::VecMul([l, r]) => todo!(),
            VecLang::VecDiv([l, r]) => todo!(),
            VecLang::VecNeg([x]) => todo!(),
            VecLang::VecSqrt(x) => todo!(),
            VecLang::VecSgn([x]) => todo!(),
            VecLang::VecMAC([a, b, c]) => todo!(),
            VecLang::Symbol(sym) => todo!(),
        }
    }

    fn make_layer(synth: &Synthesizer<Self>, iter: usize) -> Vec<Self> {
        todo!("make_layer")
    }

    fn is_valid(
        synth: &mut Synthesizer<Self>,
        lhs: &egg::Pattern<Self>,
        rhs: &egg::Pattern<Self>,
    ) -> bool {
        todo!("is_valid")
    }
}

fn main() {
    println!("test");
}
