use artha::{
    NeuralNetwork,
    logln,
    neural_net:: {
        normalize_val,
        denormalize_val,
        find_max,
        mean_loss,
    }
};
use ndarray::array;
fn main() {
    let mut xs = array![[2.,9.],[1.,5.],[3.,6.]];
    let mut new = array![[2.,9.],[1.,5.],[3.,6.]];
    let mut ys = array![[92.], [86.], [89.]];

    logln!("Input: ", xs);
    logln!("Actual Output: ", ys);

    normalize_val(&find_max(&xs), &mut xs);
    normalize_val(&vec![100.], &mut ys);

    let mut nn = NeuralNetwork::new(2,1,vec![6]);

    logln!("Training...");
    nn.train(&xs,&ys,3);
    let predicted = nn.predict(&xs);
    let loss =  mean_loss(&ys, &predicted);

    logln!("Predicted Output: ", predicted);
    logln!("Mean Square Diff: ", loss);

    logln!("Predicting new outcomes for: ", new);
    let max = find_max(&new);
    normalize_val(&max, &mut new);
    let mut result = nn.predict(&new);
    denormalize_val(&max, &mut result);
    logln!("Prediction: ", result);
}