#[macro_export]
macro_rules! debug{
    ($($x: expr),*) => {
        {
            $(
            if let Some(s) = (&$x as &std::any::Any).downcast_ref::<&str>() {
                print!("{}", s);
            } else {
                print!("{:?}", $x);
            }
            )*
            println!("");
        }
    };
}
#[macro_export]
macro_rules! log{
    ($($x: expr),*) => {
        {
            $(
                print!("{}", $x);
            )*
            println!("");
        }
    };
}

pub mod matrix;

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