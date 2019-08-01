/// Removes the need for specifying the debug format string in `print!`.
/// 
/// `debug!(var)` is a shorthand for `print("{:?}", var)`
/// 
/// _Note: the quotes around a string parameter is removed for readability_
#[macro_export]
macro_rules! debug{
    ($($x: expr),*) => {
        {
            $(
            if let Some(s) = (&$x as &std::any::Any).downcast_ref::<&str>() {
                print!("{}", s);
            } else {
                print!("{:#?}", $x);
            }
            )*
            println!("");
        }
    };
}
/// Removes the need for specifying the debug format string in `println!`.
/// 
/// `debugln!(var)` is a shorthand for `println("{:?}", var)`
/// 
/// _Note: the quotes around a string parameter is removed for readability_
#[macro_export]
macro_rules! debugln{
    ($($x: expr),*) => {
        {
            $(
            if let Some(s) = (&$x as &std::any::Any).downcast_ref::<&str>() {
                println!("{}", s);
            } else {
                println!("{:#?}", $x);
            }
            )*
        }
    };
}
/// Removes the need for specifying the display format string in `println!`.
/// 
/// `logln!(var)` is a shorthand for `println!("{}", var)`
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
/// Removes the need for specifying the display format string in `print!`.
/// 
/// `log!(var)` is a shorthand for `print!("{}", var)`
#[macro_export]
macro_rules! logln{
    ($($x: expr),*) => {
        {
            $(
                println!("{}", $x);
            )*
        }
    };
}