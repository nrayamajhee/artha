//! A dead simple Neural Network
 
use ndarray::{Array, Array1, Array2};
use pbr::ProgressBar;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

/// A dead simple Neural Network
#[derive(Debug)]
pub struct NeuralNetwork {
    hidden_sizes: Vec<usize>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl NeuralNetwork {
    /// creates a new neural network with given input, output, and hidden sizes
    pub fn new(input_size: usize, output_size: usize, hidden_sizes: Vec<usize>) -> Self {
        assert!(hidden_sizes.len() >= 1, "Can't have a network without a hidden layer!");

        let mut biases = Vec::new();
        let mut weights = vec![Array::random((input_size, hidden_sizes[0]), Uniform::new(0.,1.))];

        for i in 0..hidden_sizes.len() {
            biases.push(Array::random(hidden_sizes[i], Uniform::new(0.,1.)));
            let cols = if i == hidden_sizes.len() - 1 {
                output_size
            } else {
                hidden_sizes[i + 1]
            };
            weights.push(Array::random((hidden_sizes[i], cols), Uniform::new(0.,1.)));
        }

        Self {
            hidden_sizes,
            biases,
            weights,
        }
    }

    pub fn forward(&self, xs: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut activations = Vec::new(); 
        let mut input = xs.clone();
        for each in self.weights.iter() {
            let d = input.dot(each);
            input = sigmoid(&d);
            activations.push(input.clone());
        }
        activations
    }


    pub fn predict(&self, xs: &Array2<f64>) -> Array2<f64> {
        let mut input = xs.clone();
        for each in self.weights.iter() {
            let d = input.dot(each);
            input = sigmoid(&d);
        }
        input
    }

    // Backward propagation through the network
    pub fn backward(&mut self, xs: &Array2<f64>, ys: &Array2<f64>, activations: &Vec<Array2<f64>>) {
        let o = activations.last().expect("Forward feed produced no output!");
        let mut error = ys - o;
        for (i,each) in self.weights.iter_mut().enumerate().rev() {
            let delta = error * sigmoid_prime(o);
            error = delta.dot(&each.t());
            let z = if i > 0 {
                &activations[i-1]
            } else {
                xs
            };
            *each += &(z.t().dot(&delta));
        }
    }
    // Train the network with the given input, output n number of times
    pub fn train(&mut self, xs: &Array2<f64>, ys: &Array2<f64>, n: usize) {
        let mut pb = ProgressBar::new(n as u64);
        pb.format("[=>-]");
        for _ in 0..n {
            let activations = self.forward(xs);
            pb.inc();
            self.backward(xs, ys, &activations);
        }
    }
}

use std::fmt;

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        for i in 0..self.biases.len() {
            out = format!("{}Weight:\n{}\nBias:\n{}\n", out, self.weights[i], self.biases[i]);
        }
        out = format!("{}Weight:\n{}", out, self.weights.last().unwrap());
        write!(f, "{}", out)
    }
}

/// Loops through the Array and calculates sigmoid of each value.
///
/// _Note: Since taking the sigmoid of each value in the matrix requires it to be converted
/// to floating point and From f64 trait is not implemented for i32/64
/// nor for u32/64/size, this function returns f64 Array_
pub fn sigmoid<T>(arr: &Array2<T>) -> Array2<f64>
where
T: Copy,
f64: std::convert::From<T>,
{
    let mut new = Vec::new();
    for each in arr.iter() {
        let e = -f64::from(*each);
        new.push(1. / (1. + e.exp()));
    }
    let shape = arr.shape();
    let dim: [usize; 2] = [shape[0],shape[1]];
    Array::from_shape_vec(dim, new).unwrap()
}
/// Loops through the Array and calculates sigmoid prime of each value.
///
/// _Note: Since taking the sigmoid of each value in the matrix requires it to be converted
/// to floating point and From f64 trait is not implemented for i32/64
/// nor for u32/64/size, this function returns f64 Array_
pub fn sigmoid_prime<T>(arr: &Array2<T>) -> Array2<f64>
where
T: Copy + std::ops::Mul,
f64: std::convert::From<T>,
{
    let mut new = Vec::new();
    for each in arr.iter() {
        new.push(f64::from(*each) * (1. - f64::from(*each)));
    }
    let shape = arr.shape();
    let dim: [usize; 2] = [shape[0],shape[1]];
    Array::from_shape_vec(dim, new).unwrap()
}

/// Finds the maximum values of each column and returns them as a vector of maximum values
pub fn find_max<T: Default + std::cmp::PartialOrd + Copy>(array: &Array2<T>) -> Vec<T> {
    let dim = array.shape();
    let mut maxes: Vec<T> = Vec::new();
    for _ in 0..dim[1] {
        maxes.push(Default::default())
    }
    // println!("Each {:?}", maxes);
    for (i,each) in array.iter().enumerate() {
        if *each > maxes[i % dim[1]] {
            maxes[i % dim[1]] = *each;
        }
    }
    maxes
}

/// Divides each value in the array with the given max value
pub fn normalize_val<T>(max: &Vec<T>, array: &mut Array2<T>)
where
T: Copy + Default + std::cmp::PartialOrd + std::ops::Div + std::convert::From<f64>,
f64: std::convert::From<T>,
{
    let width = array.shape()[1];
    for (i,each) in array.iter_mut().enumerate() {
        *each = T::from(f64::from(*each) / f64::from(max[i % width]))
    }
}

/// Divides each value in the array with the given max value
pub fn denormalize_val<T>(max: &Vec<T>, array: &mut Array2<T>)
where
T: Copy + Default + std::cmp::PartialOrd + std::ops::Div + std::convert::From<f64>,
f64: std::convert::From<T>,
{
    let width = array.shape()[1];
    for (i,each) in array.iter_mut().enumerate() {
        *each = T::from(f64::from(*each) * f64::from(max[i % width]))
    }
}

/// Calculates the mean square differnce between two arrays
pub fn mean_loss<T>(a: &Array2<T>, b: &Array2<T>) -> f64 
where T: std::ops::Sub<Output = T> + Copy,
f64: std::convert::From<T>,
{
    a.iter().zip(b.iter()).fold(0.,|acc, (x,y)|{
        let diff = f64::from(*x - *y);
        acc + f64::powi(diff,2)
    }) / a.len() as f64
}