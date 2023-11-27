use ruler::enumo::Metric;
use ruler::recipe_utils::base_lang;
use ruler::recipe_utils::iter_metric;

fn main() {
    let three = iter_metric(base_lang(3), "EXPR", Metric::Atoms, 3);
    // let mut iter = three.into_iter();
    // println!("\n\n== starting ==\n");
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    // println!("\nyield {}\n", iter.next().unwrap());
    for x in three.force() {
        println!("yield {x}");
        // eprintln!("\nyield {x}\n");
    }
}
