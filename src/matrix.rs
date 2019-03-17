//! A 2D Matrix system to make our neural network implementaion easier.

use std::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
    iter::FromIterator,
    ops::{Add, AddAssign, Deref, Mul, Sub},
};

/// Error thrown when two matrix with incorrect dimensions are multiplied
///
/// For example:
/// Trying to multiply these two matrices:
/// ```
/// let x1 = matrix![[2, 3, 5],[1, 2, 8],[3, 5, 9]];
/// let x2 = matrix![[2, 3],[1, 2]];
/// log!(x1.dot(&x2).unwrap());
/// ```
/// will produce the following error:
/// ```
/// Matrices dont have the correct dimensions to perform the dot operation. The given dimensions were: 3x3 and 2x2
/// ```
#[derive(Debug)]
pub struct MatMulDimMismatchErr;
impl Error for MatMulDimMismatchErr {}
impl Display for MatMulDimMismatchErr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Matrices dont have the correct dimensions to perform the dot operation."
        )
    }
}

/// Error thrown when two matrix with different dimensions are compared
///
/// For example:
/// Trying to calculate mean squared difference between these two matrices:
/// ```
/// let x1 = matrix![[2, 3, 5],[1, 2, 8],[3, 5, 9]];
/// let x2 = matrix![[2, 3],[1, 2]];
/// log!(x1.mse_diff(&x2).unwrap());
/// ```
/// will produce the following error:
/// ```
/// Matrices dont have the correct dimensions to calculate MSE. The given dimensions were: 3x3 and 2x2
/// ```
#[derive(Debug)]
pub struct MSEDimMismatchErr;
impl Error for MSEDimMismatchErr {}
impl Display for MSEDimMismatchErr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Matrices dont have the correct dimensions to calculate MSE."
        )
    }
}

/// Error thrown when max value of an empty matrix is queried
///
/// For example:
/// Trying to find mac of the following matrix:
/// ```
/// let x1 = matrix![[2, 3],[],[3, 5]];
/// log!(x1.dot(&x2).unwrap());
/// ```
/// will produce the following error:
/// ```
/// Matrix doesn't have any elements OR has empty elements.
/// ```
#[derive(Debug)]
pub struct EmptyMatrixErr;
impl Error for EmptyMatrixErr {}
impl Display for EmptyMatrixErr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Matrix doesn't have any elements OR has empty elements."
        )
    }
}

/// A 2D generic Matrix with `Vec<Vec<T>>` as underlying data structure
#[derive(Debug)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    pub dim: (usize, usize),
}

/// Loops through the Matrix and calculates sigmoid prime of each value.
///
/// _Note: Since taking the sigmoid of each value in the matrix requires it to be converted
/// to floating point and From f64 trait is not implemented for i32/64
/// nor for u32/64/size, this function returns f64 Matrix_
pub fn sigmoid_prime<T>(current: &Matrix<T>) -> Matrix<f64>
where
    f64: std::convert::From<T>,
    T: Copy,
{
    let normalized: Vec<Vec<f64>> = current
        .data
        .iter()
        .map(|row| {
            row.iter()
                .map(|e| {
                    let e = f64::from(*e);
                    e * (1. - e)
                })
                .collect()
        })
        .collect();
    Matrix::from(normalized)
}

/// Loops through the Matrix and calculates sigmoid of each value.
///
/// _Note: Since taking the sigmoid of each value in the matrix requires it to be converted
/// to floating point and From f64 trait is not implemented for i32/64
/// nor for u32/64/size, this function returns f64 Matrix_
pub fn sigmoid<T>(current: &Matrix<T>) -> Matrix<f64>
where
    f64: std::convert::From<T>,
    T: Copy,
{
    let normalized: Vec<Vec<f64>> = current
        .data
        .iter()
        .map(|row| {
            row.iter()
                .map(|e| {
                    let e = -f64::from(*e);
                    1. / (1. + e.exp())
                })
                .collect()
        })
        .collect();
    Matrix::from(normalized)
}

/// Finds the mean squared difference of the matrix with another matrix
///
/// _Note: Since calculating MSE of each value in the matrix requires it to be converted
/// to floating point as powi is not a trait, this function returns f64 Matrices._
pub fn mse_diff<T>(one: &Matrix<T>, another: &Matrix<T>) -> Result<f64, Box<std::error::Error>>
where
    T: Sub + Clone,
    f64: std::convert::From<<T as Sub>::Output>,
{
    if one.dim == another.dim {
        let mut sum = 0.;
        for i in 0..one.dim.0 {
            for j in 0..one.dim.1 {
                let diff: f64 =
                    f64::from((one.data[i][j]).clone() - (another.data[i][j]).clone());
                sum += diff.powi(2);
            }
        }
        Ok(sum / (one.dim.0 * one.dim.1) as f64)
    } else {
        println!(
            "{} The given dimensions were: {:?} and {:?}",
            MSEDimMismatchErr, one.dim, another.dim
        );
        Err(Box::new(MSEDimMismatchErr))
    }
}

impl<T> Matrix<T> {
    /// Creates a new Matrix
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            dim: (0, 0),
        }
    }
    /// Computes the dot product of this matrix with another i.e matrix multiplication
    pub fn dot(&self, another: &Matrix<T>) -> Result<Matrix<T>, Box<Error>>
    where
        T: Mul + AddAssign + AddAssign<<T as Mul>::Output> + Copy + Default + Debug,
    {
        if self.dim.1 == another.dim.0 {
            let mut result = Vec::new();
            for i in 0..self.dim.0 {
                let mut row = Vec::new();
                for j in 0..another.dim.1 {
                    let mut r = Default::default();
                    for k in 0..another.dim.0 {
                        r += self.data[i][k] * another.data[k][j];
                    }
                    row.push(r);
                }
                result.push(row);
            }
            Ok(Matrix::from(result))
        } else {
            println!(
                "{} The given dimensions were: {:?} and {:?}",
                MatMulDimMismatchErr, self.dim, another.dim
            );
            Err(Box::new(MatMulDimMismatchErr))
        }
    }
    /// Returns the tranpose of the given Matrix
    pub fn transpose(&self) -> Matrix<T>
    where
        T: Default + Debug + Copy,
    {
        let mut transpose = Vec::new();
        for i in 0..self.dim.1 {
            let mut row = Vec::new();
            for j in 0..self.dim.0 {
                row.push(self.data[j][i]);
            }
            transpose.push(row);
        }
        Matrix::from(transpose)
    }
    /// Finds the maximum of each column in the matrix and returns them as `Vec`
    pub fn max(&self) -> Result<Vec<T>, Box<std::error::Error>>
    where
        T: std::cmp::PartialOrd + Default + Copy,
    {
        if self.data.len() == 0 {
            Err(Box::new(EmptyMatrixErr))
        } else {
            let mut maxes = Vec::new();
            for _ in 0..self.data[0].len() {
                maxes.push(Default::default())
            }
            for row in self.data.iter() {
                let len = row.len();
                if len == 0 {
                    return Err(Box::new(EmptyMatrixErr));
                } else {
                    for i in 0..len {
                        let each = row[i];
                        if each > maxes[i] {
                            maxes[i] = each;
                        }
                    }
                }
            }
            Ok(maxes)
        }
    }
}

impl<T: Copy> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        Matrix::from(self.data.clone())
    }
}

impl<T: Add<Output = T> + Copy> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(ap, bp)| *ap + *bp).collect())
            .collect()
    }
}

impl<T: Add<Output = T> + Copy> AddAssign for Matrix<T> {
    fn add_assign(&mut self, other: Matrix<T>) {
        *self = self
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(ap, bp)| *ap + *bp).collect())
            .collect()
    }
}

impl<T: Sub<Output = T> + Copy> Sub for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(ap, bp)| *ap - *bp).collect())
            .collect()
    }
}

impl<T: Mul<Output = T> + Copy> Mul for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(ap, bp)| *ap * *bp).collect())
            .collect()
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(data: Vec<Vec<T>>) -> Self {
        let dim = if data.len() > 0 {
            (data.len(), data[0].len())
        } else {
            (0, 0)
        };
        Self { data, dim }
    }
}

impl<T> Deref for Matrix<T> {
    type Target = Vec<Vec<T>>;

    fn deref(&self) -> &Vec<Vec<T>> {
        &self.data
    }
}

impl<T> FromIterator<Vec<T>> for Matrix<T> {
    fn from_iter<I: IntoIterator<Item = Vec<T>>>(iter: I) -> Self {
        let mut new = Vec::new();
        for i in iter {
            new.push(i)
        }
        Matrix::from(new)
    }
}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut form = String::new();
        for (i, row) in self.data.iter().enumerate() {
            let mut row_str = String::new();
            for (i, column) in row.iter().enumerate() {
                row_str = if i == row.len() - 1 {
                    format!("{}{:.8}", row_str, column)
                } else {
                    format!("{}{:.8}\t", row_str, column)
                }
            }
            form = if i == self.data.len() - 1 {
                format!("{}|  {}\t|", form, row_str)
            } else {
                format!("{}|  {}\t|\n", form, row_str)
            }
        }
        write!(f, "{}", form)
    }
}

/// Creates a matrix containing the arguments.
///
/// `matrix!` allows `Matrix` to be define in similary manner to `Vec` or arrays.
/// Since `Matrix` is a 2D matrix, you have to put `[]` around the inner
/// data as well.
///
/// For example:
/// ```
/// let x1 = matrix![[2., 3.],[1., 2.],[3., 5.]];
/// let x2 = matrix![[2, 3, 5],[1, 2, 5]];
/// let x3 = matrix![[3],[4],[5]];
/// ```
#[macro_export]
macro_rules! matrix {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                let mut inner_vec = Vec::new();
                for i in 0..$x.len() {
                    inner_vec.push($x[i]);
                }
                temp_vec.push(inner_vec);
            )*
            Matrix::from(temp_vec)
        }
    };
}
