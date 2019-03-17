use artha::{
    logln,
    matrix,
    matrix::{
        Matrix,
        mse_diff,
    },
    NeuralNetwork,
};
fn main() {
    let xs = matrix![[2.,9.],[1.,5.],[3.,6.]];
    let max = xs.max().unwrap();
    let xs: Matrix<f64> = xs.iter().map(|a|vec![a[0] / max[0], a[1] / max[1]]).collect();
    let ys = matrix![[92.], [86.], [89.]];
    let ys: Matrix<f64> = ys.iter().map(|a|vec![a[0] / 100.]).collect();
    let mut nn = NeuralNetwork::new(2,1,vec![3]);
    logln!("Input: ", xs);
    logln!("Actual Output: ", ys);
    let predicted = nn.train(&xs, &ys, 10000);
    logln!("Predicted Output: ", predicted);
    logln!("Loss: ", mse_diff(&predicted,&ys).unwrap());
}