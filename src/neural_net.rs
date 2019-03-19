//! A dead simple Neural Network
 
use ndarray::{Array, Array2};
use pbr::ProgressBar;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;


/// A dead simple Neural Network
#[derive(Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    output_size: usize,
    hidden_sizes: Vec<usize>,
    input_weights: Array2<f64>,
    hidden_weights: Vec<Array2<f64>>,
    z: Array2<f64>,
    dz: Array2<f64>,
}
impl NeuralNetwork {
    /// creates a new neural network with given input, output, and hidden sizes
    pub fn new(input_size: usize, output_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let input_weights = Array::random((input_size, hidden_sizes[0]), Normal::new(0.,1.));
        let mut hidden_weights = Vec::new();
        for i in 0..hidden_sizes.len() {
            let cols = if i == hidden_sizes.len() - 1 {
                output_size
            } else {
                hidden_sizes[i + 1]
            };
            let hidden_weight = Array::random((hidden_sizes[i], cols), Normal::new(0.,1.));
            hidden_weights.push(hidden_weight);
        }
        Self {
            input_size,
            output_size,
            hidden_sizes,
            hidden_weights,
            z: input_weights.clone(),
            dz: input_weights.clone(),
            input_weights,
        }
    }

    /// Forward propagation through the network
    pub fn forward(&mut self, xs: &Array2<f64>) -> Array2<f64> {
        let z = xs.dot(&self.input_weights);
        self.z = sigmoid(&z);
        let z = self.z.dot(&self.hidden_weights[0]);
        sigmoid(&z)
    }
    /// Backward propagation through the network
    pub fn backward(&mut self, xs: &Array2<f64>, ys: &Array2<f64>, o: &Array2<f64>) {
        let o_error = ys - o;
        let o_delta = o_error * sigmoid_prime(o);
        let z = o_delta.dot(&self.hidden_weights[0].t());
        self.dz = z * sigmoid_prime(&self.z);
        self.input_weights += &(xs.t().dot(&self.dz));
        self.hidden_weights[0] += &(self.z.t().dot(&o_delta));
    }
    /// Train the network with the given input, output n number of times
    pub fn train(&mut self, xs: &Array2<f64>, ys: &Array2<f64>, n: usize) -> Array2<f64> {
        let mut o = Array::zeros(ys.raw_dim());
        let mut pb = ProgressBar::new(n as u64);
        pb.format("[=>-]");
        for _ in 0..n {
            o = self.forward(xs);
            pb.inc();
            self.backward(xs, ys, &o);
        }
        o
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
pub fn normaize_val<T>(max: Vec<T>, array: &mut Array2<T>)
where
T: Copy + Default + std::cmp::PartialOrd + std::ops::Div + std::convert::From<f64>,
f64: std::convert::From<T>,
{
    let width = array.shape()[1];
    for (i,each) in array.iter_mut().enumerate() {
        *each = T::from(f64::from(*each) / f64::from(max[i % width]))
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