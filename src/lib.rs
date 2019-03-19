//! _A dead simple neural network built as a learning exercise._
//! 
//! # Getting Started 
//! ```
//! use artha::{
//!     NeuralNetwork,
//!     neural_net::{
//!         normaize_val,
//!         mean_loss,
//!         find_max,
//!     }
//! };
//! use ndarray::{array, Array2};
//! fn main() {
//!     let mut xs = array![[2.,9.],[1.,5.],[3.,6.]];
//!     normaize_val(find_max(&xs), &mut xs);
//!     let mut ys = array![[92.], [86.], [89.]];
//!     normaize_val(vec![100.], &mut ys);
//!     let mut nn = NeuralNetwork::new(2,1,vec![3]);
//!     let predicted = nn.train(&xs, &ys, 10000);
//!     let loss =  mean_loss(&ys, &predicted);
//! 
//!     use artha::logln;
//!     logln!("Input: ", xs);
//!     logln!("Actual Output: ", ys);
//!     logln!("Predicted Output: ", predicted);
//!     logln!("Loss: ", loss);
//! }
//! ```
//! 
//! This program is a direct translation of <https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python>
//! into rust.
//! 
//! Also checko 3Blue1Browns's excellent series on Neural Network <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi> 
//! 
//! I found this network to be significantly slower than the tutorial. Perhaps ndarray is not as fast as numpy or
//! perhaps my rust code is not optimiz. I'll definitely look into it.
//!
//! Besides optimization, I am also hoping to implmenting a network that recognized handwritten digits and who knows what else from there.
//! But for now, this is a fairly inaccurate rookie version that I could build on my own.
//! 
//! - If you have any questions or suggestions, feel free to submit issues, or contact me in other ways.
//! - If you found my sub-par rust skills offensive, please do provide some constructive criticism.
//!

mod log_macros;
pub mod neural_net;
pub use self::neural_net::NeuralNetwork;
