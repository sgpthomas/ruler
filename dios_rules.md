# Attempt 1: with timeout
```
rw!("name"; "(- (- ?a ?b) (- ?c ?d))" <=> "(- (- ?a ?c) (- ?b ?d))")
rw!("name"; "(VecAdd (VecAdd ?a ?b) (VecAdd ?c ?d))" <=> "(VecAdd (VecAdd ?d ?a) (VecAdd ?c ?b))")
rw!("name"; "(+ (+ ?a ?b) (+ ?c ?d))" <=> "(+ (+ ?c ?a) (+ ?b ?d))")
rw!("name"; "(* (* ?a ?b) (* ?c ?d))" <=> "(* (* ?a ?c) (* ?b ?d))")
rw!("name"; "(VecMinus (VecMinus ?a ?b) (VecAdd ?c ?d))" <=> "(VecMinus (VecMinus ?a ?c) (VecAdd ?b ?d))")
rw!("name"; "(VecMinus (VecAdd ?a ?b) (VecMinus ?c ?d))" <=> "(VecMinus (VecAdd ?a ?d) (VecMinus ?c ?b))")
rw!("name"; "(VecMinus ?a (VecMinus ?b (VecAdd ?c ?d)))" <=> "(VecAdd (VecAdd ?d ?a) (VecMinus ?c ?b))")
rw!("name"; "(VecMinus (VecAdd ?a ?b) (VecAdd ?c ?d))" <=> "(VecAdd ?b (VecMinus (VecMinus ?a ?c) ?d))")
rw!("name"; "(VecMinus (VecMinus ?a ?b) (VecMinus ?c ?d))" <=> "(VecAdd (VecMinus ?a ?b) (VecMinus ?d ?c))")
rw!("name"; "(- (- ?a ?b) (- ?c ?d))" <=> "(+ (- ?a ?b) (- ?d ?c))")
rw!("name"; "(VecAdd (Vec ?a ?b) (Vec ?c ?d))" <=> "(VecAdd (Vec ?c ?b) (Vec ?a ?d))")
rw!("name"; "(- (+ ?a ?b) (- ?c ?d))" <=> "(+ (- ?a ?c) (+ ?d ?b))")
rw!("name"; "(- (+ ?a ?b) (- ?c ?d))" <=> "(- (+ ?a ?d) (- ?c ?b))")
rw!("name"; "(VecAdd (Vec ?a ?b) (Vec ?c ?d))" <=> "(Vec (+ ?a ?c) (+ ?b ?d))")
rw!("name"; "(- (* ?a ?b) (- ?c ?d))" <=> "(+ (* ?a ?b) (- ?d ?c))")
rw!("name"; "(VecMinus (Vec ?a ?b) (Vec ?c ?d))" <=> "(Vec (- ?a ?c) (- ?b ?d))")
rw!("name"; "(VecAdd ?a (VecAdd ?b ?c))" <=> "(VecAdd ?b (VecAdd ?a ?c))")
rw!("name"; "(* ?a (* ?b ?c))" <=> "(* ?b (* ?a ?c))")
rw!("name"; "(+ ?a (+ ?b ?c))" <=> "(+ ?b (+ ?a ?c))")
rw!("name"; "(VecMinus (VecAdd ?a ?b) ?c)" <=> "(VecAdd ?b (VecMinus ?a ?c))")
rw!("name"; "(- ?a (+ ?b ?c))" <=> "(- (- ?a ?c) ?b)")
rw!("name"; "(VecMinus ?a (VecAdd ?b ?c))" <=> "(VecMinus (VecMinus ?a ?c) ?b)")
rw!("name"; "(VecMinus ?a (VecMinus ?b ?c))" <=> "(VecAdd ?a (VecMinus ?c ?b))")
rw!("name"; "(VecAdd ?a (VecMinus ?b ?c))" <=> "(VecAdd ?b (VecMinus ?a ?c))")
rw!("name"; "(- ?a (- ?b ?c))" <=> "(+ ?c (- ?a ?b))")
rw!("name"; "(- ?a (- ?b ?c))" <=> "(- (+ ?a ?c) ?b)")
rw!("name"; "(VecMinus (VecMinus ?a ?b) (VecMinus ?c ?b))" => "(VecMinus ?a ?c)")
rw!("name"; "(VecAdd (VecAdd ?a ?b) (VecMinus ?c ?c))" => "(VecAdd ?a ?b)")
rw!("name"; "(VecAdd ?a (VecMinus ?b (VecAdd ?b ?c)))" => "(VecMinus ?a ?c)")
rw!("name"; "(+ ?a (- ?b (+ ?a ?c)))" => "(- ?b ?c)")
rw!("name"; "(+ ?a (+ ?b (- ?c ?b)))" => "(+ ?a ?c)")
rw!("name"; "(VecMinus ?a (VecMinus (VecMinus ?a ?b) ?c))" => "(VecAdd ?b ?c)")
rw!("name"; "(VecAdd ?a (VecMinus (VecAdd ?b ?c) ?b))" => "(VecAdd ?a ?c)")
rw!("name"; "(+ ?a (- (- ?b ?a) ?c))" => "(- ?b ?c)")
rw!("name"; "(VecAdd ?a (VecMinus (VecMinus ?b ?b) ?c))" => "(VecMinus ?a ?c)")
rw!("name"; "(- (* ?a ?b) (* ?a ?c))" <=> "(* ?a (- ?b ?c))")
rw!("name"; "(+ (* ?a ?b) (* ?a ?c))" <=> "(* ?a (+ ?b ?c))")
rw!("name"; "(* (- ?a ?b) (- ?c ?b))" <=> "(* (- ?b ?c) (- ?b ?a))")
rw!("name"; "(* (- ?a ?b) (- ?b ?c))" <=> "(* (- ?c ?b) (- ?b ?a))")
rw!("name"; "(- ?a (+ ?b (+ ?a ?c)))" => "(- ?b (+ ?c (+ ?b ?b)))")
rw!("name"; "(VecMinus (Vec ?a ?b) (Vec ?c ?b))" => "(VecMinus (Vec ?a ?c) (Vec ?c ?c))")
rw!("name"; "(Vec (Vec ?a ?b) (VecMinus ?c ?c))" => "(Vec (Vec ?a ?b) (VecMinus ?a ?a))")
rw!("name"; "(Concat (VecMinus ?a ?b) (VecMinus ?c ?c))" => "(Concat (VecMinus ?a ?b) (VecMinus ?a ?a))")
rw!("name"; "(Vec (VecMinus ?a ?b) (VecMinus ?c ?c))" => "(Vec (VecMinus ?a ?b) (VecMinus ?a ?a))")
rw!("name"; "(Concat (Vec ?a ?b) (VecMinus ?c ?c))" => "(Concat (Vec ?a ?b) (VecMinus ?a ?a))")
rw!("name"; "(Vec (Concat ?a ?b) (VecMinus ?c ?c))" => "(Vec (Concat ?a ?b) (VecMinus ?a ?a))")
rw!("name"; "(+ ?a (+ ?b (* 0 ?c)))" => "(+ ?a ?b)")
rw!("name"; "(* ?a (- ?b ?c))" <=> "(* -1 (* ?a (- ?c ?b)))")
rw!("name"; "(+ ?a (* ?b ?c))" <=> "(+ 0 (+ ?a (* ?b ?c)))")
rw!("name"; "(+ 0 (* ?a (* ?b ?c)))" <=> "(* ?a (* ?b ?c))")
rw!("name"; "(VecAdd "<[Int(0), Int(0)]>" (Vec ?a (* ?b ?c)))" <=> "(Vec ?a (* ?b ?c))")
rw!("name"; "(VecMinus (Vec ?a ?b) (Vec 0 ?c))" <=> "(Vec ?a (- ?b ?c))")
rw!("name"; "(VecAdd (Vec 0 ?a) (Vec ?b ?c))" <=> "(Vec ?b (+ ?a ?c))")
rw!("name"; "(Vec (- ?a ?b) ?c)" <=> "(Vec (- ?a ?b) (+ 0 ?c))")
rw!("name"; "(- (* ?a ?b) ?c)" <=> "(+ (* ?a ?b) (* -1 ?c))")
rw!("name"; "(Vec (* ?a ?b) ?c)" <=> "(Vec (* ?a ?b) (+ 0 ?c))")
rw!("name"; "(VecAdd (Vec ?a 0) (Vec ?b ?c))" <=> "(Vec (+ ?b ?a) ?c)")
rw!("name"; "(VecMinus (Vec ?a ?b) (Vec ?a ?c))" => "(Vec (* 0 ?c) (- ?b ?c))")
rw!("name"; "(* (- ?a ?b) (- -1 ?c))" <=> "(* (- ?b ?a) (+ 1 ?c))")
rw!("name"; "(+ ?a ?b)" <=> "(+ ?b ?a)")
rw!("name"; "(VecAdd ?a ?b)" <=> "(VecAdd ?b ?a)")
rw!("name"; "(* ?a ?b)" <=> "(* ?b ?a)")
rw!("name"; "(Concat ?a (VecMinus ?b ?b))" => "(Concat ?a (VecMinus ?a ?a))")
rw!("name"; "(VecAdd ?a (VecMinus ?b ?a))" => "(VecMinus (VecAdd ?b ?b) ?b)")
rw!("name"; "(Vec ?a (VecMinus ?b ?b))" => "(Vec ?a (VecMinus ?a ?a))")
rw!("name"; "(VecMinus ?a (VecAdd ?a ?b))" => "(VecMinus ?b (VecAdd ?b ?b))")
rw!("name"; "(- (* ?a ?a) (* ?b ?b))" <=> "(* (+ ?a ?b) (- ?a ?b))")
rw!("name"; "(+ ?a ?b)" <=> "(+ 0 (+ ?a ?b))")
rw!("name"; "(* ?a ?b)" <=> "(* ?a (+ 0 ?b))")
rw!("name"; "(- ?a ?b)" <=> "(+ ?a (* -1 ?b))")
rw!("name"; "(- ?a (* ?a ?b))" <=> "(* ?a (- 1 ?b))")
rw!("name"; "(- ?a ?a)" <=> "(* 0 ?a)")
rw!("name"; "(- ?a 1)" <=> "(+ -1 ?a)")
rw!("name"; "(- ?a -1)" <=> "(+ 1 ?a)")
rw!("name"; "(- ?a 0)" <=> "(+ 0 ?a)")
rw!("name"; "(+ 0 ?a)" <=> "(* 1 ?a)")
rw!("name"; "(- 0 ?a)" <=> "(* -1 ?a)")
rw!("name"; "(+ -1 (* ?a ?a))" <=> "(* (+ -1 ?a) (+ 1 ?a))")
```

