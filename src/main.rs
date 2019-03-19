use artha::{
    NeuralNetwork,
    neural_net::{
        normaize_val,
        mean_loss,
        find_max,
    }
};
use ndarray::{array, Array2};
fn main() {
    let mut xs = array![[2.,9.],[1.,5.],[3.,6.]];
    normaize_val(find_max(&xs), &mut xs);
    let mut ys = array![[92.], [86.], [89.]];
    normaize_val(vec![100.], &mut ys);
    let mut nn = NeuralNetwork::new(2,1,vec![3]);
    let predicted = nn.train(&xs, &ys, 10000);
    let loss =  mean_loss(&ys, &predicted);

    use artha::logln;
    logln!("Input: ", xs);
    logln!("Actual Output: ", ys);
    logln!("Predicted Output: ", predicted);
    logln!("Loss: ", loss);
}