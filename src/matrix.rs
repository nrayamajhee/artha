use std::{error::Error, fmt};

#[derive(Debug)]
struct DimensionMismatch;
impl Error for DimensionMismatch {}
impl fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrices don't have the correct dimensions to perform the dot operation.")
    }
}

#[derive(Debug)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    dim: (usize, usize),
}
impl<T> Matrix<T> {
    pub fn dot(&self, another: &Matrix<T>) -> Result<Matrix<T>, Box<std::error::Error>>
    where T: std::ops::Mul + std::ops::AddAssign + std::ops::AddAssign<<T as std::ops::Mul>::Output> + Clone + Default {
        if self.dim.0 == another.dim.1 {
            let mut result = Vec::new();
            for i in 0..self.dim.1 {
                let mut row = Vec::new();
                for j in 0..another.dim.0 {
                    let mut r = Default::default();
                    for k in 0..another.dim.1 {
                        r += self.data[i][k].clone() * another.data[k][j].clone();
                    }
                    row.push(r);
                }
                result.push(row);
            }
            Ok(Matrix::from(result))
        } else {
            println!("{} The give dimensions were: {}x{} and {}x{}", DimensionMismatch, self.dim.0, self.dim.1, another.dim.0, another.dim.1);
            Err(Box::new(DimensionMismatch))
        }
    }
    pub fn sigmoid(&self) -> Matrix<T> where f64: std::convert::From<T> , T: std::convert::From<f64> + Clone {
        let normalized: Vec<Vec<T>> = self.data.iter().map(|row|{
            row.iter().map(|e|{
                let e = -f64::from((*e).clone());
                T::from(1. / (1. + e.exp()))
            }).collect()
        }).collect();
        Matrix::from(normalized)
    }
}
impl<T> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(data: Vec<Vec<T>>) -> Matrix<T> {
        Matrix {
            dim: (data[0].len(), data.len()),
            data,
        }
    }
}
impl<T> fmt::Display for Matrix<T>
where T: std::fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut form = String::new();
        for (i, row) in self.data.iter().enumerate() {
            let mut row_str = String::new();
            for (i, column) in row.iter().enumerate() {
                row_str = if i == row.len() - 1 {
                    format!("{}{}", row_str, column)
                } else {
                    format!("{}{}\t", row_str, column)
                }
            }
            form = if i == self.data.len() - 1 {
                format!("{}|\t{}\t|", form, row_str)
            } else {
                format!("{}|\t{}\t|\n", form, row_str)
            }
        }
        write!(f, "{}", form)
    }
}