use artha::{
    debug,
    log,
    matrix,
    matrix::Matrix,
};
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
fn merge_vec(a:&Vec<f64>, b: &Vec<f64>) -> Vec<Vec<f64>> {
        let smaller = if a.len() != b.len() {
            println!("Two vectors are not equal, merging up to the smaller one!");
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
            c.push(vec![a[i], b[i]]);
        }
        c
}
fn scalar_dot(a: &Vec<f64>, b: f64) -> f64 {
    a.iter().map(|x| x * b).sum()
}
fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x,y)| x * y).sum()
}
fn mse_loss(y_true: &Vec<f64>, y_false: &Vec<f64>) -> f64 {
    let sum: f64 = y_true.iter().zip(y_false.iter()).map(|(x,y)| f64::powi(x - y, 2)).sum();
    sum / y_true.len() as f64
}
struct NeuralNetwork{
    input_size: usize,
    output_size: usize,
    hidden_sizes: Vec<usize>,
    input_weights: Vec<Vec<f64>>,
    hidden_weights: Vec<Vec<f64>>,
    z: Vec<Vec<f64>>,
}
use rand::prelude::*;
impl NeuralNetwork {
    fn new(input_size: usize, output_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut input_weights = Vec::new();
        let mut rng = rand::thread_rng();
        for i in 0..input_size {
            let mut input = Vec::new();
            for _ in 0..hidden_sizes[0] {
                input.push(rng.gen())
            }
            input_weights.push(input);
        }
        let mut hidden_weights = Vec::new();
        for i in 0..hidden_sizes.len() {
            if i == hidden_sizes.len() - 1 {
                let mut hidden = Vec::new();
                for _ in 0..output_size * hidden_sizes[i] {
                    hidden.push(rng.gen())
                }
                hidden_weights.push(hidden);
            } else {
                let mut hidden = Vec::new();
                for _ in 0..hidden_sizes[i] * hidden_sizes[i+1] {
                    hidden.push(rng.gen())
                }
                hidden_weights.push(hidden);
            }
        }
        Self {
            input_size,
            output_size,
            hidden_sizes,
            input_weights,
            hidden_weights,
            z: Vec::new(),
        }
    }
    fn forward(&mut self, xs: Vec<Vec<f64>>) -> f64 {
        // let z = matrix_dot(&xs, &self.input_weights).unwrap();
        // self.z = sigmoid_matrix(z);
        // let z3 = matrix_dot(&self.z, &self.input_weights);
        // let z2 = scalar_dot(self.z, self.hidden_weights[0])
        0.
    }
}
fn main() {
    let x1 = matrix![[2, 3],[1, 2],[3, 5]];
    let x2 = matrix![[2, 3, 5],[1, 2, 5]];
    log!(x1.dot(&x2).unwrap());
    log!(x1.sigmoid());
    // let x2 = vec![0.,5.,6.];
    // let max1 = find_max(&x1).unwrap();
    // let max2 = find_max(&x2).unwrap();
    // let x1: Vec<f64> = x1.iter().map(|a|a/max1).collect();
    // let x2: Vec<f64> = x2.iter().map(|a|a/max1).collect();
    // let xs = merge_vec(&x1, &x2);
    // let ys = vec![92., 86., 89.];
    // let ys: Vec<f64> = ys.iter().map(|a|a/100.).collect();
    // let nn = NeuralNetwork::new(2,1,vec![3]);
}