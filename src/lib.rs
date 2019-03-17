//! _A dead simple neural network built as a learning exercise._
//! 
//! # Getting Started 
//! ```
//!     let xs = matrix![[2.,9.],[1.,5.],[3.,6.]];
//!     let max = xs.max().unwrap();
//!     let xs: Matrix<f64> = xs.iter().map(|a|vec![a[0] / max[0], a[1] / max[1]]).collect();
//!     let ys = matrix![[92.], [86.], [89.]];
//!     let ys: Matrix<f64> = ys.iter().map(|a|vec![a[0] / 100.]).collect();
//!     let mut nn = NeuralNetwork::new(2,1,vec![3]);
//!     logln!("Input: ", xs);
//!     logln!("Actual Output: ", ys);
//!     let predicted = nn.train(&xs, &ys, 100000);
//!     logln!("Predicted Output: ", predicted);
//!     logln!("Loss: ", predicted.mse_diff(&ys).unwrap());
//! ```
//! 
//! This program is a direct translation of <https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python>
//! into rust.
//! 
//! Due to my naive custom Matrix implementation, this network is significantly slower that the tutorial.
//! I'll definitely look into optimizing matrix opeartions and other segments of my code.
//! Also checkout 3Blue1Browns's excellent series on Neural Network <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi> 
//!
//! Besides optimization, I am also hoping to implmenting a network that recognized handwritten digits and who knows what else from there.
//! But for now, this is a fairly inaccurate rookie version that I could build on my own.
//! 
//! - If you have any questions or suggestions, feel free to submit issues, or contact me in other ways.
//! - If you found my sub-par rust skills offensive, please do provide some constructive criticism.
//!

mod log_macros;
pub mod matrix;
mod neural_net;

// pub use self::matrix::EmptyMatrixErr;
// pub use self::matrix::MSEDimMismatchErr;
// pub use self::matrix::MatMulDimMismatchErr;
pub use self::matrix::Matrix;
pub use self::neural_net::NeuralNetwork;
