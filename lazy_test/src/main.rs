use ruler::enumo::{Metric, Workload};
use ruler::recipe_utils::base_lang;
use ruler::recipe_utils::iter_metric;

fn main() {
    let base = base_lang(3);
    let base_alt = base.clone().plug("EXPR", &Workload::new(["expr"]));
    println!("{base:#?}");
    // let three = iter_metric(base, "EXPR", Metric::Atoms, 4);
    let wkld = base.clone().plug("EXPR", &base_alt);
    // let mut iter = wkld.into_iter();

    for sexp in wkld {
        println!("yield {sexp}");
    }

    // let three = Workload::new(["(op A A)"])
    //     .plug("A", &Workload::new(["0", "1"]))
    //     .plug("op", &Workload::new(["+", "-"]));
    // let mut iter = three.into_iter();
    // println!("\n\n== starting ==\n");
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    // for x in three.force() {
    //     println!("yield {x}");
    //     // eprintln!("\nyield {x}\n");
    // }
}
