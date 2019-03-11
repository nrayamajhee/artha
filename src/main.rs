fn find_max(list: &Vec<f64>) -> Result<f64, Box<std::error::Error>>{
    let first = list.iter().next().ok_or("Need at least one input!")?;
    let max = list.iter().try_fold(first, |acc, x| {
        let cmp = x.partial_cmp(acc)?;
        let max = if let std::cmp::Ordering::Greater = cmp {
            x
        } else {
            acc
        };
        Some(max)
    });
    Ok(*max.unwrap())
}
fn merge_vec(a:&Vec<f64>, b: &Vec<f64>) -> Vec<[f64;2]> {
        let smaller = if a.len() != b.len() {
            println!("Two vectors are not equal, merging smaller section!");
            if a.len() > b.len() {
                b
            } else {
                a
            }
        } else {
            a
        };
        let mut c = Vec::new();
        for i in 0..smaller.len() {
            c.push([a[i], b[i]]);
        }
        c
}
fn main() {
    let x1 = vec![2.,1.,3.];
    let x2 = vec![0.,5.,6.];
    let max1 = find_max(&x1).unwrap();
    let max2 = find_max(&x2).unwrap();
    let x1: Vec<f64> = x1.iter().map(|a|a/max1).collect();
    let x2: Vec<f64> = x2.iter().map(|a|a/max1).collect();
    let xs = merge_vec(&x1, &x2);
    let ys = vec![92., 86., 89.];
    let ys: Vec<f64> = ys.iter().map(|a|a/100.).collect();
    let nn = NeuralNetwork::new(2,1,3);
}

struct NeuralNetwork{
    input_size: usize,
    output_size: usize,
    hidden_size: usize,
    input_weights: Vec<f64>,
    hidden_weights: Vec<f64>,
}
impl NeuralNetwork {
    pub fn new(input_size: usize, output_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            hidden_size,
        }
    }
}

