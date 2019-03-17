use crate::matrix::{
    Matrix,
    sigmoid,
    sigmoid_prime,
};
use pbr::ProgressBar;

/// A dead simple Neural Network
#[derive(Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    output_size: usize,
    hidden_sizes: Vec<usize>,
    input_weights: Matrix<f64>,
    hidden_weights: Vec<Matrix<f64>>,
    z: Matrix<f64>,
    dz: Matrix<f64>,
}
use rand::Rng;
impl NeuralNetwork {
    /// creates a new neural network with given input, output, and hidden sizes
    pub fn new(input_size: usize, output_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut i_weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..input_size {
            let mut row: Vec<f64> = Vec::new();
            for _ in 0..hidden_sizes[0] {
                row.push(rng.gen())
            }
            i_weights.push(row);
        }
        let input_weights = Matrix::from(i_weights);
        let mut hidden_weights = Vec::new();
        for i in 0..hidden_sizes.len() {
            let cols = if i == hidden_sizes.len() - 1 {
                output_size
            } else {
                hidden_sizes[i + 1]
            };
            let mut hidden = Vec::new();
            for _ in 0..hidden_sizes[i] {
                let mut row: Vec<f64> = Vec::new();
                for _ in 0..cols {
                    row.push(rng.gen())
                }
                hidden.push(row);
            }
            hidden_weights.push(Matrix::from(hidden));
        }
        Self {
            input_size,
            output_size,
            hidden_sizes,
            input_weights,
            hidden_weights,
            z: Matrix::new(),
            dz: Matrix::new(),
        }
    }
    /// Forward porpagation through the network
    pub fn forward(&mut self, xs: &Matrix<f64>) -> Matrix<f64> {
        let z = xs.dot(&self.input_weights).unwrap();
        self.z = sigmoid(&z);
        let z = self.z.dot(&self.hidden_weights[0]).unwrap();
        sigmoid(&z)
    }
    /// Backward porpagation through the network
    pub fn backward(&mut self, xs: Matrix<f64>, ys: Matrix<f64>, o: Matrix<f64>) {
        let o_prime = sigmoid_prime(&o);
        let o_error = ys - o;
        let o_delta = o_error * o_prime;
        let z = o_delta.dot(&self.hidden_weights[0].transpose()).unwrap();
        self.dz = z * sigmoid_prime(&self.z);
        self.input_weights += xs.transpose().dot(&self.dz).unwrap();
        self.hidden_weights[0] += self.z.transpose().dot(&o_delta).unwrap();
    }
    /// Train the network with the given input, output n number of times
    pub fn train(&mut self, xs: &Matrix<f64>, ys: &Matrix<f64>, n: usize) -> Matrix<f64> {
        let mut o = Matrix::new();
        let mut pb = ProgressBar::new(n as u64);
        pb.format("[=>-]");
        for _ in 0..n {
            o = self.forward(xs);
            pb.inc();
            self.backward(xs.clone(), ys.clone(), o.clone());
        }
        o
    }
}
