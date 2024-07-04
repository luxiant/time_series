use anyhow::Error;
use ndarray::Array2;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use statrs::{distribution::{ContinuousCDF, FisherSnedecor}, statistics::Statistics};
use rand::prelude::{SeedableRng, StdRng};
use rand_distr::{Distribution, Normal};
use serde_json;
// use serde::Deserialize;
// use std::io;
// use std::io::Read;

pub mod durbin_watson_test {
    use std::f64::consts::PI;
    use ndarray::{prelude::*, OwnedRepr, Array, Array1, Array2};
    use ndarray_linalg::{solve::Inverse, eig::Eig, qr::QR, trace::Trace};
    use statrs::distribution::{Normal, ContinuousCDF};
    use nalgebra::DMatrix;

    // need to convert all of these inputs into rust polars dataframe
    pub fn gradsol(x: &f64, a: Array1<&f64>, m: i32, n: i32) -> f64 {
        let mut sum0: f64;
        let d: i32;
        let mut j1: i32;
        let mut _j2: i32;
        let mut j3: i32;
        let mut j4: i32;
        let mut _k: i32;
        let mut _h: i32;
    
        let mut nu: i32 = a.iter().position(|&ai| ai > x).unwrap_or(m as usize) as i32;
        if nu == m {
            sum0 = 1.0;
        } else if nu == 0 {
            sum0 = 0.0;
        } else {
            let mut k: i32 = 1;
            let mut h: i32 = m - nu;
            if h >= nu {
                h = nu;
                k = -k;
                d = 2;
                j1 = 0;
                _j2 = 2;
                j3 = 3;
                j4 = 1;
            } else {
                nu += 1;
                d = -2;
                j1 = m-2;
                _j2 = m-1;
                j3 = m+1;
                j4 = m;
            };
    
            let pin: f64 = PI / (2.0 as f64 * n as f64);
            sum0 = (k + 1) as f64 / (2.0 as f64);
            let mut sgn0 = k as f64 / n as f64;
            let n2: i32 = 2 * n - 1;
    
            // first integral
            let st: i32 = (h - (2 * ((h / 2) as i32))) as i32;
            for _l1 in (0..=st).rev() {
                // take into account that k = -np.sign(d)
                for l2 in (0..=(nu - k)).step_by(d as usize) {
                    let sum1: &f64 = a[(j4 - 1) as usize];
                    let prod0: &f64 = if l2 == 0 {
                        x
                    } else {
                        a[l2 as usize - 1]
                    };
                    let u: f64 = 0.5 * (sum1 + prod0);
                    let v: f64 = 0.5 * (sum1 - prod0);
                    let mut sum1: f64 = 0.0;
                    for i in (1..=n2).step_by(2) {
                        let y: f64 = (u - v * (i as f64 * pin).cos()) as f64;
                        let num: f64 = y - x;
                        let prod1: f64 = a.slice(s![..j1]).iter().map(|&ai| num / (y - ai)).product();
                        let prod2: f64 = a.slice(s![j3 - 1..]).iter().map(|&ai| num / (y - ai)).product();
                        let prod = &(prod1 * prod2);
                        sum1 += (prod.abs()).sqrt();
                    }
                    sgn0 = -sgn0;
                    sum0 += sgn0 * sum1;
                    j1 += d;
                    j3 += d;
                    j4 += d;
                }
                // second integral
                if d == 2 {
                    j3 -= 1;
                } else {
                    j1 += 1;
                }
                _j2 = 0;
                nu = 0;
            }
        }
    
        sum0
    }
    
    pub fn dw_stat(res: Array2<f64>) -> f64 {
        let diff = &(res.t().slice(s![.., 1..])) - &(res.t().slice(s![.., ..-1]));
        let sum_diff_sq = diff.mapv(|x| x.powi(2)).sum();
        let sum_res_sq = res.mapv(|x| x.powi(2)).sum();
    
        (sum_diff_sq / sum_res_sq) as f64
    }
    
    pub fn dwtest_pan(res: Array2<f64>, x: Array2<f64>, matrix_inverse: bool, n: i32) -> (f64, f64) {
        let dw = dw_stat(res.clone());
        let (n_obs, dim) = (x.nrows(), x.ncols());
        let mut upp_one = Array::zeros((n_obs, n_obs));
        for i in 0..(n_obs-1) {
            upp_one[[i, i+1]] = 1.0;
        }
        let eye: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array::eye(n_obs);
        let mut a_matrix = (Array::eye(n_obs) * 2.0) - upp_one.clone() - upp_one.t();
        a_matrix[[0, 0]] = 1.0;
        a_matrix[[n_obs-1, n_obs-1]] = 1.0;
    
        let q_qt_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
        if matrix_inverse {
            let gram_matrix_x = x.t().dot(&x);
            let r_matrix = gram_matrix_x.inv().unwrap();
            q_qt_matrix = x.dot(&r_matrix).dot(&x.t());
        } else {
            let q_matrix = x.qr().unwrap().0;
            q_qt_matrix = q_matrix.dot(&q_matrix.t());
        };
    
        let a_qqt: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = a_matrix.clone() - q_qt_matrix.dot(&a_matrix) + eye;
        let mut eigvals_real = a_qqt.eig().unwrap().0.mapv(|x| x.re).to_vec();
        eigvals_real.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut eigvals_real_second_process = eigvals_real.iter().map(|x| x - 1.0).collect::<Vec<f64>>();
        let eigvals_real_third_process = eigvals_real_second_process.split_off(dim);
        let eigvals_result = Array1::from(eigvals_real_third_process.iter().filter(|&x| x > &1e-10).collect::<Vec<&f64>>());
    
        let p_right = gradsol(&dw, eigvals_result.clone(), eigvals_result.len() as i32, n);
        
        (p_right, dw)
    }
    
    pub fn dw_meanvar(x: Array2<f64>, matrix_inverse: bool) -> (f64, f64) {
        let (n_obs, dim) = (x.nrows(), x.ncols());
        let mut ax_matrix = Array2::<f64>::zeros((n_obs, dim));
    
        // Construct AX matrix
        ax_matrix.row_mut(0).assign(&(&x.row(0) - &x.row(1)));
        ax_matrix.row_mut(n_obs - 1).assign(&(&x.row(n_obs - 1) - &x.row(n_obs - 2)));
        
        for i in 1..n_obs-1 {
            ax_matrix.row_mut(i).assign(&(&x.row(i) * 2.0 - &x.row(i-1) - &x.row(i+1)));
        }
    
        // Calculate AXR matrix
        let axr_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = if matrix_inverse {
            let r_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = x.t().dot(&x).inv().unwrap();
            ax_matrix.dot(&r_matrix)
        } else {
            let a_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = x.t().dot(&x);
            let ax_t_matrix = ax_matrix.t();
            // get axr that makes sqaure of a_matrix * axr = ax_matrix.t() least.
            let a_matrix_dmatrix = DMatrix::from_iterator(a_matrix.nrows(), a_matrix.ncols(), a_matrix.iter().map(|x| *x));
            let ax_t_dmatrix_dmatrix = DMatrix::from_iterator(ax_t_matrix.nrows(), ax_t_matrix.ncols(), ax_t_matrix.iter().map(|x| *x));
            let a_matrix_dmatrix_pseudo_inverse = a_matrix_dmatrix.pseudo_inverse(1e-6).unwrap();
            let axr_dmatrix = a_matrix_dmatrix_pseudo_inverse * ax_t_dmatrix_dmatrix;
            Array2::from_shape_fn((axr_dmatrix.nrows(), axr_dmatrix.ncols()), |(i, j)| axr_dmatrix[(i, j)])
        };
    
        // Calculate B matrix
        let b_matrix = x.t().dot(&axr_matrix);
        let p_matrix_value = 2.0 * (n_obs as f64 - 1.0) - b_matrix.trace().unwrap();
        let q_matrix_value = 2.0 * (3.0 * n_obs as f64 - 4.0) - 2.0 * (ax_matrix.t().dot(&axr_matrix)).trace().unwrap() + b_matrix.dot(&b_matrix).trace().unwrap();
        let dw_mean = p_matrix_value / (n_obs as f64 - dim as f64);
        let dw_var = 2.0 / ((n_obs as f64 - dim as f64) * (n_obs as f64 - dim as f64 + 2.0)) * (q_matrix_value - p_matrix_value * dw_mean);
    
        (dw_mean, dw_var)
    }
    
    pub fn dwtest(res: Array2<f64>, x: Array2<f64>, tail: &str, method: &str, matrix_inverse: bool, n: i32) -> (f64, f64) {
        let (mut p_right, dw) = if method == "pan" {
            let (p_right, dw) = dwtest_pan(res.clone(), x.clone(), matrix_inverse, n);
            if p_right < 0.0 || p_right > 1.0 {
                // warning: Pan's method did not work
                (-1.0, dw)
            } else {
                (p_right, dw)
            }
        } else if method == "normal" {
            (dw_stat(res.clone()), -1.0)
        } else {
            // error: no such method
            (-1.0, -1.0)
        };
    
        if p_right < 0.0 {
            // use normal approximation
            let (dw_mean, dw_var) = dw_meanvar(x.clone(), matrix_inverse);
            let normal = Normal::new(0.0, 1.0).unwrap();
            p_right = normal.cdf((dw - dw_mean) / dw_var.sqrt());
        }
    
        let p_val = match tail {
            "both" => 2.0 * p_right.min(1.0 - p_right),
            "right" => p_right,
            "left" => 1.0 - p_right,
            _ => -1.0, // warning: no such test
        };
    
        (p_val, dw)
    }
}

pub mod find_p_d_q {
    use unit_root::prelude::distrib::{AlphaLevel, Regression};
    use unit_root::prelude::*;
    use ndarray::Array2;
    
    pub fn find_adequate_diff(data: Array2<f64>, start_diff: i32) -> i32 {
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let mut diff = start_diff;
        let y = nalgebra::DVector::from_row_slice(&data_slice);
        let regression = Regression::Constant;
        let report = tools::adf_test(&y, diff as usize, regression).unwrap();
        let critical_value = distrib::dickeyfuller::get_critical_value(regression, report.size, AlphaLevel::FivePercent).unwrap();
        let t_stat = report.test_statistic;
        if t_stat >= critical_value {
            diff += 1;
            find_adequate_diff(data, diff)
        } else {
            diff
        }
    }

    pub fn find_adequate_p(data: Array2<f64>) -> i32 {
        let len_data = data.nrows();
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let pacfs = arima::pacf(data_slice, Some(len_data)).unwrap();
        let mut max_p: i32 = 0;
        let mut max_pacf: f64 = pacfs[0];
        for i in 0..pacfs.len() {
            if pacfs[i].abs() > max_pacf.abs() {
                max_pacf = pacfs[i];
                max_p = i as i32;
            }
        }
        
        max_p
    }

    pub fn find_adequate_q(data: Array2<f64>) -> i32 {
        let len_data = data.nrows();
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let acfs = arima::acf(data_slice, Some(len_data), true).unwrap();
        let mut max_q: i32 = 0;
        for i in 0..acfs.len() {
            if acfs[i] <= 0.0 {
                break;
            } else {
                max_q = i as i32;
            }
        }

        max_q
    }
}

pub mod arima {
    use anyhow::Result;
    use std::cmp;
    use rand::distributions::{Distribution, Normal};
    use rand::{thread_rng, Rng};
    use ndarray::{Array1, Array2};
    use ndarray_linalg::Inverse;
    use ndarray_stats::SummaryStatisticsExt;

    pub fn acf(data: &[f64], max_lag: Option<usize>, covariance: bool) -> Result<Vec<f64>> {
        let max_lag = match max_lag {
            Some(max_lag) => cmp::min(max_lag, data.len() - 1),
            None => data.len() - 1,
        };
        let acf_total_len = max_lag + 1;

        let data_len = data.len();
        let data_len_clone = data_len.clone();
        let mut sum = 0.0;
        for i in 0..data_len {
            sum += data[i]
        }
        let data_mean = sum / (data_len_clone as f64);

        let mut y = vec![0.0; acf_total_len];
        for m in 0..acf_total_len {
            for i in 0..(data_len-m) {
                let data_len_clone = data_len.clone();
                let i_th_deviation = data[i] - data_mean;
                let i_t_th_deviation = data[i+m] - data_mean;
                y[m] += (i_th_deviation * i_t_th_deviation) / (data_len_clone as f64);
            }
            if !covariance && m > 0 {
                y[m] = y[m]/y[0];
            }
        }

        Ok(y)
    }

    pub fn pacf(data: &[f64], max_lag: Option<usize>) -> Result<Vec<f64>> {
        let max_lag = match max_lag {
            Some(max_lag) => cmp::min(max_lag, data.len() - 1),
            None => data.len() - 1,
        };

        let max_lag_clone = max_lag.clone();
        let mut pacf_values = Array1::zeros(max_lag_clone+1);
        pacf_values[0] = 1.0;

        for lag in 1..max_lag {
            let mut x = Array2::<f64>::zeros((lag, lag));
            let mut y = Array1::<f64>::zeros(lag);
    
            for i in 0..lag {
                for j in 0..lag {
                    x[(i, j)] = data[(lag - 1 - i) as usize];
                }
                y[i] = data[lag + i];
            }
    
            let x_inv = x.inv().unwrap();
            let coef = x_inv.dot(&y);
            pacf_values[lag] = coef[lag - 1];
        }

        let result_pacf = pacf_values.into_raw_vec();

        Ok(result_pacf)
    }

    pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
        let data_clone = data.clone();
        let mut diff = vec![0.0; data_clone.len()-d];
        for i in d..data_clone.len() {
            diff[i-d] = data[i] - data[i-d];
        }

        diff
    }

    pub fn fit_arima(data: &[f64], p: usize, d: usize, q: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
        let differenced_data = if d == 0 as usize {
            data.clone().to_vec()
        } else {
            difference(data, d).as_slice().to_vec()
        };

        let mut fitted = true;

        let max_lag = if p >= q {
            p.clone()
        } else {
            q.clone()
        };

        // initialize coef and residuals for durbin recursion
        let mut residuals_hat = vec![1.0;differenced_data.len()-max_lag];

        let mut data_sum = 0.0;
        for i in 0..differenced_data.len() {
            data_sum += differenced_data[i];
        }
        let intercept = data_sum/(differenced_data.len() as f64);

        let mut ar_coef = vec![1.0; p];
        let mut ma_coef = vec![1.0; q];

        // durbin's recursion method
        let max_iter = 100000;
        let tol = 1e-6;

        let mut data_hat = vec![0.0; differenced_data.len()-max_lag];
        
        for i in 0..max_iter {
            for t in max_lag..differenced_data.len() {
                let mut x_t = 0.0;
                for ar_i in 0..ar_coef.len() {
                    x_t += ar_coef[ar_i]*differenced_data[t-ar_i-1];
                }
                for ma_i in 0..ma_coef.len() {
                    x_t += ma_coef[ma_i]*residuals_hat[t-ma_i-1];
                }
                x_t += intercept;
                data_hat[t] = x_t;
                residuals_hat[t] = differenced_data[t]-x_t;
            }

            let mut new_ar_coef = vec![0.0; ar_coef.len()];
            let mut new_ma_coef = vec![0.0; ma_coef.len()];
            for i in 0..ar_coef.len() {
                let numerator_phi: f64 = (i+1..differenced_data.len()).map(|t| differenced_data[t] * differenced_data[t-i-1]).sum();
                let denominator_phi: f64 = (i+1..differenced_data.len()).map(|t| differenced_data[t-i-1] * differenced_data[t-i-1]).sum();
                new_ar_coef[i] = numerator_phi / denominator_phi;
            }
            for i in 0..ma_coef.len() {
                let numerator_theta: f64 = (i+1..differenced_data.len()).map(|t| residuals_hat[t] * residuals_hat[t-i-1]).sum();
                let denominator_theta: f64 = (i+1..differenced_data.len()).map(|t| residuals_hat[t-i-1] * residuals_hat[t-i-1]).sum();
                new_ma_coef[i] = numerator_theta / denominator_theta;
            }

            let ar_coef_diff: f64 = new_ar_coef.iter().zip(&ar_coef).map(|(e, p)| (e - p).abs()).sum();
            let ma_coef_diff: f64 = new_ma_coef.iter().zip(&ma_coef).map(|(e, p)| (e - p).abs()).sum();
            if ar_coef_diff.abs() < tol && ma_coef_diff.abs() < tol {
                break;
            } else {
                ar_coef = new_ar_coef.clone();
                ma_coef = new_ma_coef.clone();
                if i == max_iter-1 {
                    fitted = false;
                }
            }
        }

        let mut result_coef: Vec<f64> = Vec::new();
        result_coef.push(intercept);
        for i in 0..ar_coef.len() {
            result_coef.push(ar_coef[i]);
        }
        for i in 0..ma_coef.len() {
            result_coef.push(ma_coef[i]);
        }

        let mut hat_undifferentiated = vec![0.0; data_hat.len()+d];
        for i in 0..d {
            hat_undifferentiated[i] = data[i];
        }
        for i in 0..data_hat.len() {
            hat_undifferentiated[i+d] = hat_undifferentiated[i] + data_hat[i];
        }

        (result_coef, hat_undifferentiated, residuals_hat, fitted)
    }

    pub fn forecast(simulated_data: &[f64], residuals_hat: &[f64], coef: Vec<f64>, p: usize, d: usize, q: usize, simulation_length: usize) -> Vec<f64> {
        let mut differenced_data = if d == 0 as usize {
            simulated_data.clone().to_vec()
        } else {
            difference(simulated_data, d).as_slice().to_vec()
        };
        let mut residuals_hat_clone = residuals_hat.clone().to_vec();
        let residual_vec_for_stat = Array1::from(residuals_hat_clone);
        let residual_mean = residual_vec_for_stat.mean().unwrap();
        let residual_std = residual_vec_for_stat.std(0.0);

        let mut residuals_hat_clone = residuals_hat.clone().to_vec();
        let intercept = coef[0];
        let ar_coef = coef[1..1+p].to_vec();
        let ma_coef = coef[1+p..1+p+q].to_vec();

        for t in 0..simulation_length {
            let mut x_t = 0.0;
            for ar_i in 0..ar_coef.len() {
                x_t += ar_coef[ar_i]*differenced_data[differenced_data.len()+t-ar_i-1];
            }
            for ma_i in 0..ma_coef.len() {
                x_t += ma_coef[ma_i]*residuals_hat_clone[differenced_data.len()+t-ma_i-1];
            }
            x_t += intercept;
            differenced_data.push(x_t);

            let normal = Normal::new(residual_mean, residual_std);
            let mut rng = thread_rng();
            let random_number = normal.sample(&mut rng);
            residuals_hat_clone.push(random_number);
        }

        let mut hat_undifferentiated = vec![0.0; differenced_data.len()+d];
        for i in 0..d {
            hat_undifferentiated[i] = data[i];
        }
        for i in 0..differenced_data.len() {
            hat_undifferentiated[i+d] = hat_undifferentiated[i] + differenced_data[i];
        }

        let result = hat_undifferentiated[hat_undifferentiated.len()-simulation_length..];

        result
    }

    // pub fn fit_ar(data: &[f64], p: usize) -> Result<Vec<f64>> {
    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();
    //     let mut x = Array2::<f64>::zeros(((data_len-p), p));
    //     let mut y = Array1::<f64>::zeros(data_len-p);

    //     for i in 0..(data_len - p) {
    //         for j in 0..p {
    //             x[(i, j)] = data[i + j];
    //         }
    //         y[i] = data[i + p];
    //     }
    
    //     let svd_solver_result = x.least_squares(&y).unwrap();
    //     let mut coef = svd_solver_result.solution.into_raw_vec();

    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();
    //     let coef_clone = coef.clone();
    //     let p_clone = p.clone();
    //     let y_ar_hat = ar_simulation(data_clone, coef_clone, p_clone);
    //     let mut residual_containing_intercept: Vec<f64> = Vec::with_capacity(data_len-p);
    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();
    //     for i in 0..data_len {
    //         residual_containing_intercept.push(data[i] - y_ar_hat[i]);
    //     }
    //     let get_residuals_in_only_simulated_part = residual_containing_intercept[p..].to_vec();
        
    //     let mut sum = 0.0;
    //     for i in 0..get_residuals_in_only_simulated_part.len() {
    //         sum += get_residuals_in_only_simulated_part[i]
    //     }
    //     let intercept = sum/(get_residuals_in_only_simulated_part.len() as f64);
    //     coef.push(intercept);

    //     Ok(coef)
    // }

    // pub fn fit_ar(data: &[f64], p: usize) -> Vec<f64> {
    //     let mut r: Vec<f64> = Vec::new();
    //     let mut sum_of_squares_data = 0.0;
    //     for i in 0..data.len() {
    //         sum_of_squares_data += data[i] * data[i];
    //     }
    //     let mean_of_squares_data = sum_of_squares_data / (data.len() as f64);

    //     for j in 1..(p+1) {
    //         let mut sum_squares = 0.0;
    //         for i in 0..(data.len()-j) {
    //             sum_squares += data[i]*data[i+j];                
    //         }
    //         let mean_squares = sum_squares/(data.len() as f64);
    //         r[j] = mean_squares;
    //     }

    //     let r_slice = r[..p].to_vec();

    //     let mut vector_form_toeplitz_matrix: Vec<Vec<f64>> = Vec::new();
    //     for i in 0..p {
    //         let insert = vec![0.0; p];
    //         vector_form_toeplitz_matrix.push(insert);
    //     }

    //     for i in 0..p {
    //         for j in 0..p {
    //             if j >= i {
    //                 vector_form_toeplitz_matrix[i][j] = r_slice[j-i];
    //             } else {
    //                 vector_form_toeplitz_matrix[i][j] = r_slice[i-j];
    //             }
    //         }
    //     }

    //     let r_vec = r[1..p+1].to_vec();
        
    //     // cholesky LDLT decomposition
    //     fn signum(value: f64) -> i32 {
    //         if value > 0.0 {
    //             1
    //         } else if value < 0.0 {
    //             -1
    //         } else {
    //             0
    //         }
    //     }

    //     fn cholesky_ldlt_decomposition(matrix: &Vec<Vec<f64>>, max_condition_number: f64) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    //         let n = matrix.len();
    //         let mut chol_d = vec![0.0; n];
    //         let mut chol_l = vec![vec![0.0; n]; n];
        
    //         let mut current_max = -1.0;
        
    //         for j in 0..n {
    //             let mut val = 0.0;
    //             for k in 0..j {
    //                 val += chol_d[k] * chol_l[j][k] * chol_l[j][k];
    //             }
    //             let mut diag_temp = matrix[j][j] - val;
    //             let diag_sign = signum(diag_temp);
        
    //             if diag_sign == 0 && max_condition_number < -0.5 {
    //                 return None;
    //             }

    //             if max_condition_number > -0.5 {
    //                 if current_max <= 0.0 {
    //                     if diag_sign == 0 {
    //                         diag_temp = 1.0;
    //                     }
    //                 } else {
    //                     if diag_sign == 0 {
    //                         diag_temp = (current_max / max_condition_number).abs();
    //                     } else {
    //                         if (diag_temp * max_condition_number).abs() < current_max {
    //                             diag_temp = diag_sign as f64 * (current_max / max_condition_number).abs();
    //                         }
    //                     }
    //                 }
    //             }
        
    //             chol_d[j] = diag_temp;
    //             if diag_temp.abs() > current_max {
    //                 current_max = diag_temp.abs();
    //             }
    //             chol_l[j][j] = 1.0;
        
    //             for i in (j+1)..n {
    //                 val = 0.0;
    //                 for k in 0..j {
    //                     val += chol_d[k] * chol_l[j][k] * chol_l[i][k];
    //                 }
    //                 chol_l[i][j] = (matrix[i][j] - val) / chol_d[j];
    //             }
    //         }
        
    //         Some((chol_d, chol_l))
    //     }

    //     let (chol_d, chol_l) = cholesky_ldlt_decomposition(&vector_form_toeplitz_matrix, 100.0).unwrap();

    //     let mut y: Vec<f64> = vec![0.0; p];
    //     let mut yule_walker: Vec<f64> = r_vec.clone();
        
    //     for i in 0..p {
    //         let mut val = 0.0;
    //         for j in 0..i {
    //             val += chol_l[i][j]*y[j];
    //         }
    //         y[i] = yule_walker[i] - val;
    //     }

    //     for i in (0..p-1).rev() {
    //         let mut val = 0.0;
    //         for j in i+1..p {
    //             val += chol_l[i][j]*yule_walker[j];
    //         }
    //         yule_walker[i] = (yule_walker[i]/chol_d[i]) - val;
    //     }

    //     yule_walker
    // }

    // pub fn ar_simulation(data: &[f64], coef: Vec<f64>, lag: usize) -> Vec<f64> {
    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();

    //     let mut starting_data = data[0..lag].to_vec();
        
    //     fn serial_recursion(starting_data: &mut Vec<f64>, coef: Vec<f64>, simulation_len: usize) -> &mut Vec<f64> {
    //         let data_clone = starting_data.clone();
    //         let data_len = data_clone.len();
    //         let coef_clone = coef.clone();
    //         let p = coef_clone.len();
            
    //         let mut hat = 0.0;
    //         for i in 0..p {
    //             hat += coef[i]*starting_data[data_len-i];
    //         }
    //         starting_data.push(hat);

    //         let starting_data_clone = starting_data.clone();
    //         let starting_data_len = starting_data_clone.len();
    //         if starting_data_len == simulation_len {
    //             starting_data
    //         } else {
    //             let simulated_data = serial_recursion(starting_data, coef, simulation_len);
    //             simulated_data
    //         }
    //     }

    //     let simulated_hat = serial_recursion(&mut starting_data, coef, data_len);

    //     simulated_hat.clone()
    // }

    // pub fn fit_ma(data: &[f64], errors: &[f64], q: usize) -> Result<Vec<f64>> {
    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();
    //     let mut x = Array2::<f64>::zeros(((data_len-q), q));
    //     let mut y = Array1::<f64>::zeros(data_len-q);

    //     for i in 0..(data_len - q) {
    //         for j in 0..q {
    //             x[(i, j)] = data[i + j];
    //         }
    //         y[i] = data[i + q];
    //     }

    //     let svd_solver_result = x.least_squares(&y).unwrap();
    //     let coef = svd_solver_result.solution.into_raw_vec();

    //     Ok(coef)
    // }

    // pub fn ma_simulation(data: &[f64], errors: &[f64], coef: Vec<f64>, lag: usize) -> Vec<f64> {
    //     let data_clone = data.clone();
    //     let data_len = data_clone.len();

    //     let mut starting_data = data[0..lag].to_vec();

    //     let coef_clone = coef.clone();
    //     let q = coef_clone.len();

    //     let mut simulated_hat: Vec<f64> = Vec::new();
    //     for _ in lag..data_len {
    //         let mut hat = 0.0;
    //         for k in 0..q {
    //             hat += coef[k]*data[data_len-q+k]
    //         }
    //         simulated_hat.push(hat);
    //     }

    //     simulated_hat
    // }
}

pub mod garch {
    use anyhow::Error;
    use rand::Rng;
    // use rustimization::lbfgsb_minimizer::Lbfgsb;
    // use finitediff::FiniteDiff;
    use argmin::core::{CostFunction, State, Executor};
    use argmin::solver::particleswarm::ParticleSwarm;

    pub fn mean(x: &Vec<f64>) -> f64 {
        let n = x.len() as f64;

        x.iter().sum::<f64>() / n
    }
    
    pub fn std(x: &Vec<f64>) -> f64 {
        let u = mean(x);
        let sqdevs: Vec<f64> = x.iter().map(|v| (v - u).powf(2.)).collect();

        mean(&sqdevs).sqrt()
    }

    pub fn garch_recursion(omega: f64, alpha: &[f64], beta: &[f64], eps: &Vec<f64>) -> Vec<f64> {
        let mut sigma_2 = Vec::with_capacity(eps.len());
        let m = mean(eps);
        let init_sigma_2 = eps.iter().fold(0., |acc, e| acc + (e-m).powf(2.))/(eps.len()-1) as f64;
    
        for i in 0..eps.len() {
            if i < alpha.len() || i < beta.len() {
                sigma_2.push(init_sigma_2);
            } else {
                let next = predict_next(omega, alpha, beta, &eps[..i].to_vec(), &sigma_2[..i].to_vec());
                sigma_2.push(next);
            }
        }

        sigma_2
    }

    pub fn predict_next(omega: f64, alpha: &[f64], beta: &[f64], eps: &Vec<f64>, sigma_2: &Vec<f64>) -> f64 {
        let n_e = eps.len();
        let n_s = sigma_2.len();
        let residual_term = alpha.iter().enumerate().fold(0., |acc, (j, a)| {
            let t = n_e - (j + 1);
            acc + (a * eps[t].powf(2.))
        });
        let volatility_term = beta.iter().enumerate().fold(0., |acc, (j, b)| {
            let t = n_s - (j + 1);
            acc + (b * sigma_2[t])
        });

        omega + residual_term + volatility_term
    }

    pub fn neg_loglikelihood(sigma_2: &Vec<f64>, eps: &Vec<f64>) -> f64 {
        let loglik = sigma_2.iter().zip(eps).fold(0., |acc, (sig2, ep)| {
            acc + (-sig2.ln() - (ep.powf(2.)/sig2))
        });

        -loglik
    }

    pub fn fit(ts: &Vec<f64>, p: usize, q: usize) -> Option<Vec<f64>> {
        let mean = mean(ts);
        let eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();
    
        struct StructForGarchFit {
            p: usize,
            q: usize,
            eps: Vec<f64>,
        }

        impl CostFunction for StructForGarchFit {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, coef: &Self::Param) -> Result<Self::Output, Error> {
                let coef_vec = coef.clone();
                let omega = coef_vec[0];
                let alpha = coef_vec[1..((self.p+1) as usize)].to_vec();
                let beta = coef_vec[((self.p+1) as usize)..(self.p+self.q+1)].to_vec();
                let eps_clone = self.eps.clone();
                let sigma_2 = garch_recursion(omega, &alpha, &beta, &eps_clone);
                let eps_clone = self.eps.clone();
                let cost = neg_loglikelihood(&sigma_2, &eps_clone);

                Ok(cost)
            }
        }

        let p_clone = p.clone();
        let q_clone = q.clone();
        let cost = StructForGarchFit {
            p: p_clone,
            q: q_clone,
            eps: eps,
        };

        let lower_border = vec![-1.0; 1+p+q];
        let upper_border = vec![1.0; 1+p+q];
        let solver = ParticleSwarm::new((lower_border, upper_border), 10000);
    
        let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(300))
        .run().unwrap();

        let coef_option = result.state().get_best_param().unwrap().into();
        match coef_option {
            Some(x) => {
                let x_clone = x.clone();
                let result_coef = x_clone.position;
                Some(result_coef)
            },
            None => None,
        }
    }

    pub fn forecast<F: Fn(usize, &mut T) -> f64, T: Rng>(ts: &Vec<f64>, n: usize, omega: f64, alpha: &[f64], beta: &[f64], noise: &F, rng: &mut T) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let mean = mean(ts);
        let mut eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();
    
        // Initialize sigma_2 for the history we have
        let mut sigma_2 = garch_recursion(omega, alpha, beta, &eps);
    
        // Forecast
        for i in 0..n {
            let next_sigma_2 = predict_next(omega, alpha, beta, &eps, &sigma_2);
            sigma_2.push(next_sigma_2);
            let next_eps = next_sigma_2.sqrt() * noise(i, rng);
            eps.push(next_eps);
        }
    
        // Remove residuals from original time series
        eps.drain(0..ts.len());
        sigma_2.drain(0..ts.len());

        Ok((eps, sigma_2))
    }
}

fn breusch_pagan_test(res: Array2<f64>, x: Array2<f64>) -> f64 {
    let x_clone = x.clone().into_raw_vec();
    let res_power = res.clone().mapv(|x| x.powi(2)).into_raw_vec();
    let (n_obs, dim) = (x.nrows(), x.ncols());
    let data = vec![("Y", res_power), ("X1", x_clone)];
    let reg_data = RegressionDataBuilder::new().build_from(data).unwrap();
    let formula = "Y ~ X1";
    let model = FormulaRegressionBuilder::new()
    .data(&reg_data)
    .formula(formula).fit().unwrap();
    let ssr = model.ssr();
    let x_clone = x.clone().into_raw_vec();
    let y_hat = model.predict(vec![("X1", x_clone)]).unwrap();
    let y_hat_array2 = Array2::from_shape_vec((n_obs, 1), y_hat).unwrap();
    let error = &res - &y_hat_array2;
    let sse = error.mapv(|x| x.powi(2)).sum();
    let f_value = (ssr / dim as f64) / (sse / (n_obs as f64 - dim as f64 - 1.0));
    let dfn = dim as f64;
    let dfd = n_obs as f64 - dim as f64 - 1.0;
    let f_dist = FisherSnedecor::new(dfn, dfd).unwrap();
    let p_value = 1.0 - f_dist.cdf(f_value);

    p_value
}

fn check_autocorrelation_and_heteroskedascity_of_residuals(model: linregress::RegressionModel, x: Array2<f64>) -> (bool, bool) {
    let res = model.residuals();
    let res_len = res.len();
    let res_array = Array2::from_shape_vec((res_len, 1), res.to_vec()).unwrap();
    let res_array_clone = res_array.clone();
    let x_clone = x.clone();    
    let bp_value = breusch_pagan_test(res_array_clone, x_clone);

    let intercept = model.parameters()[0];
    let mut x_with_intercept = x.clone();
    for i in 0..x_with_intercept.nrows() {
        x_with_intercept[[i, 0]] = intercept;
    }
    let res_array_clone = res_array.clone();
    let x_with_intercept_clone = x_with_intercept.clone();
    let (p_val, _dw) = durbin_watson_test::dwtest(res_array_clone, x_with_intercept_clone, "both", "normal", false, 15);
    let res_array_clone = res_array.clone();
    let x_with_intercept_clone = x_with_intercept.clone();
    let (p_val_inv, _dw_inv) = durbin_watson_test::dwtest(res_array_clone, x_with_intercept_clone, "both", "normal", true, 15);

    let is_it_heteroskedastic = bp_value < 0.05;
    let is_it_autocorrelated = p_val <= 0.05 || p_val_inv <= 0.05;

    (is_it_heteroskedastic, is_it_autocorrelated)
}

fn akaike_information_criterion(num_params: f64, data: Array2<f64>) -> f64 {
    let sample_num = data.clone().nrows() as f64;
    let variance = data.clone().variance();
    let mean = data.clone().mean();
    let pi = std::f64::consts::PI;
    let log_likelihood = (-0.5 * sample_num * (2.0 * pi * variance).ln()) - ((0.5 * (1.0/variance))*(data.mapv(|x| (x - mean).powi(2)).sum()));
    let aic = (-2.0 * log_likelihood) + (2.0 * num_params);

    aic
}

fn fit_residuals_arima(residuals_array: Array2<f64>, p: i32, d: i32, q: i32) -> Result<(f64, Vec<f64>, Array2<f64>), Error> {
    let residuals_array_vec = residuals_array.clone().into_raw_vec();
    let residuals_slice = residuals_array_vec.as_slice();
    let (coef, hat_undifferentiated, residuals_hat, fitted) = arima::fit_arima(residuals_slice, p as usize, d as usize, q as usize);
    
    match coef {
        Ok(coef) => {
            let num_params = coef.len() as f64;
            let aic = akaike_information_criterion(num_params, residuals_array.clone());
            let mut rng: StdRng = SeedableRng::from_seed([0; 32]);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let resid_hat = sim::arima_sim(
                residuals_array.len(),
                Some(&coef[1..((p+1) as usize)]),
                if q > 0 {Some(&coef[((p+1) as usize)..((p+q) as usize)])} else {None},
                d as usize,
                &|mut rng| normal.sample(&mut rng),
                &mut rng,
            );
            match resid_hat {
                Ok(resid_hat) => {
                    let resid_hat_array2 = Array2::from_shape_vec((resid_hat.len(), 1), resid_hat).unwrap();
                    Ok((aic, coef, resid_hat_array2))
                },
                Err(e) => Err(e)
            }
        },
        Err(e) => Err(e)
    }
}

fn fit_residuals_garch(residuals_array: Array2<f64>, p: i32, q: i32) -> Option<(f64, Vec<f64>, Array2<f64>)>{
    let residuals_array_vec = residuals_array.clone().into_raw_vec();
    let residuals_slice = residuals_array_vec.clone();
    let coef_option = garch::fit(&residuals_slice, p as usize, q as usize);
    match coef_option {
        Some(coef) => {
            let num_params = coef.len() as f64;
            let aic = akaike_information_criterion(num_params, residuals_array.clone());
            let omega = coef[0];
            let alpha = &coef[1..(p+1) as usize];
            let beta = &coef[(p+1) as usize..];
            let mut rng: StdRng = SeedableRng::from_seed([0; 32]);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let residuals_slice = residuals_array_vec.clone();
            let (resid_hat, _sigma_2) = garch::forecast(&residuals_slice, residuals_slice.len(), omega, alpha, beta, &|_, mut rng| normal.sample(&mut rng), &mut rng).unwrap();
            let resid_hat_array2 = Array2::from_shape_vec((resid_hat.len(), 1), resid_hat).unwrap();
            
            Some((aic, coef, resid_hat_array2))
        },
        None => None,
    }
}

fn fit_residuals(rediduals_array: Array2<f64>, is_it_autocorrelated: bool, is_it_heteroskedastic: bool) -> (String, Vec<f64>, Vec<i32>, Array2<f64>) {
    let residuals_array = rediduals_array.clone();
    let mut result_fitted_residuals: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
    let model_type: String;
    let mut coef: Vec<f64> = vec![];
    let mut p_d_q = vec![];
    if is_it_autocorrelated && !is_it_heteroskedastic {
        model_type = "arima".to_string();
        let residuals_array_clone = residuals_array.clone();
        let adequate_p = find_p_d_q::find_adequate_p(residuals_array_clone);
        let residuals_array_clone = residuals_array.clone();
        let adequate_diff = find_p_d_q::find_adequate_diff(residuals_array_clone, 0);
        let residuals_array_clone = residuals_array.clone();
        let adequate_q = find_p_d_q::find_adequate_q(residuals_array_clone);

        let mut best_q = 0;
        let best_aic = f64::INFINITY;
        for i in 0..(adequate_q+1) {
            let residuals_array_clone = residuals_array.clone();
            match fit_residuals_arima(residuals_array_clone, adequate_p, adequate_diff, i) {
                Ok((aic, get_coef, resid_hat)) => {
                    if aic < best_aic {
                        best_q = i;
                        result_fitted_residuals = resid_hat;
                        coef = get_coef;
                    } else {
                        continue;
                    }
                },
                Err(_) => continue
            }
        }

        p_d_q.push(adequate_p);
        p_d_q.push(adequate_diff);
        p_d_q.push(best_q);
    } else if !is_it_autocorrelated && is_it_heteroskedastic {
        model_type = "garch".to_string();
        let residuals_array_clone = residuals_array.clone();
        let adequate_p = find_p_d_q::find_adequate_p(residuals_array_clone);
        let residuals_array_clone = residuals_array.clone();
        let adequate_q = find_p_d_q::find_adequate_q(residuals_array_clone);

        let mut best_q = 0;
        let best_aic = f64::INFINITY;
        for i in 0..(adequate_q+1) {
            let residuals_array_clone = residuals_array.clone();
            match fit_residuals_garch(residuals_array_clone, adequate_p, i) {
                Some((aic, get_coef, resid_hat)) => {
                    if aic < best_aic {
                        best_q = i;
                        result_fitted_residuals = resid_hat;
                        coef = get_coef;
                    } else {
                        continue;
                    }
                },
                None => continue
            }
        }

        p_d_q.push(adequate_p);
        p_d_q.push(0);
        p_d_q.push(best_q);
    } else {
        let residuals_array_clone = residuals_array.clone();
        let adequate_diff = find_p_d_q::find_adequate_diff(residuals_array_clone, 0);
        let residuals_array_clone = residuals_array.clone();
        let adequate_p = find_p_d_q::find_adequate_p(residuals_array_clone);
        let residuals_array_clone = residuals_array.clone();
        let adequate_q = find_p_d_q::find_adequate_q(residuals_array_clone);
        
        let mut best_q_arima = 0;
        let mut best_fit_residuals_arima: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
        let mut best_arima_coef: Vec<f64> = vec![];
        let best_aic_arima = f64::INFINITY;
        for i in 0..(adequate_q+1) {
            let residuals_array_clone = residuals_array.clone();
            match fit_residuals_arima(residuals_array_clone, adequate_p, adequate_diff, i) {
                Ok((aic, arima_coef, resid_hat)) => {
                    if aic < best_aic_arima {
                        best_q_arima = i;
                        best_arima_coef = arima_coef;
                        best_fit_residuals_arima = resid_hat;
                    } else {
                        continue;
                    }
                },
                Err(_) => continue
            }
        }

        let mut best_q_garch = 0;
        let mut best_fit_residuals_garch: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
        let mut best_garch_coef: Vec<f64> = vec![];
        let best_aic_garch = f64::INFINITY;
        for i in 0..(adequate_q+1) {
            let residuals_array_clone = residuals_array.clone();
            match fit_residuals_garch(residuals_array_clone, adequate_p, i) {
                Some((aic, garch_coef, resid_hat)) => {
                    if aic < best_aic_garch {
                        best_q_garch = i;
                        best_garch_coef = garch_coef;
                        best_fit_residuals_garch = resid_hat;
                    } else {
                        continue;
                    }
                },
                None => continue
            }
        }

        if best_aic_arima < best_aic_garch {
            result_fitted_residuals = best_fit_residuals_arima;
            model_type = "arima".to_string();
            coef = best_arima_coef;
            p_d_q.push(adequate_p);
            p_d_q.push(adequate_diff);
            p_d_q.push(best_q_arima);
            println!("adequate_p_d_q: {:?}", p_d_q);
        } else {
            result_fitted_residuals = best_fit_residuals_garch;
            model_type = "garch".to_string();
            coef = best_garch_coef;
            p_d_q.push(adequate_p);
            p_d_q.push(0);
            p_d_q.push(best_q_garch);
            println!("adequate_p_d_q: {:?}", p_d_q);
        }
    }

    (model_type, coef, p_d_q, result_fitted_residuals)
}

fn simulate_time_series(input_sample: Array2<f64>, future_forcast_length: i32) -> Array2<f64> {
    let input_sample_clone = input_sample.clone();
    let time_lapse_without_future: Vec<f64> = (0..input_sample_clone.nrows() as i32).map(|x| x as f64).collect();
    let time_lapse_with_future: Vec<f64> = (0..(input_sample.nrows() as i32 + future_forcast_length) as i32).map(|x| x as f64).collect();
    let input_sample_clone = input_sample.clone();
    let input_sample_vec = input_sample_clone.into_raw_vec();
    let time_regression_data: Vec<(&str, Vec<f64>)> = vec![("Y", input_sample_vec), ("X1", time_lapse_without_future)];
    let time_reg_data = RegressionDataBuilder::new().build_from(time_regression_data).unwrap();
    let time_formula = "Y ~ X1";
    let time_model = FormulaRegressionBuilder::new()
    .data(&time_reg_data)
    .formula(time_formula).fit().unwrap();
    let time_coef = time_model.parameters();
    let mut time_series_hat = vec![];
    for i in input_sample.len()..time_lapse_with_future.len() {
        let time_series_hat_value = time_coef[0] + time_coef[1] * time_lapse_with_future[i];
        time_series_hat.push(time_series_hat_value);
    }
    let time_series_hat_array2 = Array2::from_shape_vec((time_series_hat.len(), 1), time_series_hat).unwrap();
    let mut result_simulated_time_series = time_series_hat_array2.clone();

    let time_residuals = time_model.residuals().to_vec();
    let time_residuals_clone = time_residuals.clone();
    let time_residuals_array2 = Array2::from_shape_vec((time_residuals_clone.len(), 1), time_residuals_clone).unwrap();
    let time_residuals_clone = time_residuals_array2.clone();
    let (is_it_autocorrelated, is_it_heteroskedastic) = check_autocorrelation_and_heteroskedascity_of_residuals(time_model, time_residuals_clone);
    let time_residuals_clone = time_residuals_array2.clone();
    let (resid_model_type, resid_coef, resid_p_d_q, _fitted_residuals) = fit_residuals(time_residuals_clone, is_it_autocorrelated, is_it_heteroskedastic);
    println!("resid_model_type: {:?}", resid_model_type);
    println!("resid_coef: {:?}", resid_coef);

    let time_residuals_clone = time_residuals.clone();
    match resid_model_type.as_str() {
        "arima" => {
            let mut rng: StdRng = SeedableRng::from_seed([0; 32]);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let resid_forecast = arima_forecast(
                time_residuals_clone.as_slice(),
                future_forcast_length as usize,
                Some(resid_coef[1..((resid_p_d_q[0] + 1) as usize)].to_vec().as_slice()),
                Some(resid_coef[((resid_p_d_q[0] + 1) as usize)..].to_vec().as_slice()),
                resid_p_d_q[1] as usize,
                &|_: usize, mut rng| normal.sample(&mut rng),
                &mut rng,
            ).unwrap();
            let resid_forecast_array2 = Array2::from_shape_vec((resid_forecast.len(), 1), resid_forecast).unwrap();
            result_simulated_time_series = result_simulated_time_series + resid_forecast_array2;

            result_simulated_time_series
        },
        "garch" => {
            let mut rng: StdRng = SeedableRng::from_seed([0; 32]);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let resid_forecast = garch::forecast(
                &time_residuals_clone,
                future_forcast_length as usize,
                resid_coef[0],
                resid_coef[1..(resid_p_d_q[0] as usize + 1)].to_vec().as_slice(),
                resid_coef[(resid_p_d_q[0] as usize + 1)..].to_vec().as_slice(),
                &|_: usize, mut rng| normal.sample(&mut rng),
                &mut rng,
            ).unwrap().0;
            let resid_forecast_array2 = Array2::from_shape_vec((resid_forecast.len(), 1), resid_forecast).unwrap();
            result_simulated_time_series = result_simulated_time_series + resid_forecast_array2;

            result_simulated_time_series
        }
        _ => {
            result_simulated_time_series
        }
    }
}

fn main() {
    // #[derive(Deserialize)]
    // struct InputData {
    //     input: Vec<f64>,
    //     future_forcast_length: i32
    // }
    // let mut input_json = String::new();
    // io::stdin().read_to_string(&mut input_json).expect("Failed to read input JSON");
    // let input_data: InputData = serde_json::from_str(&input_json).expect("Failed to parse input JSON");
    // let input = input_data.input;
    // let input_data_array2 = Array2::from_shape_vec((input.len(), 1), input).unwrap();
    // let future_forcast_length = input_data.future_forcast_length;

    let input = vec![
        23250000000.0,
        23566000000.0,
        27230000000.0,
        22857000000.0,
        23724000000.0,
        24271000000.0,
        29018000000.0,
        24607000000.0,
        26666000000.0,
        26157000000.0,
        29486000000.0,
        24673000000.0,
        25783000000.0,
        24747000000.0,
        27671000000.0,
        23408000000.0,
        24924000000.0,
        23338000000.0,
        27385000000.0,
        22236000000.0,
        24047000000.0,
        22397000000.0,
        24113000000.0,
        19590000000.0,
        20813000000.0,
        19280000000.0,
        22059000000.0,
        18684000000.0,
        20238000000.0,
        19226000000.0,
        21770000000.0,
        18155000000.0,
        19289000000.0,
        19153000000.0,
        22542000000.0,
        19072000000.0,
        20003000000.0,
        18756000000.0,
        21761000000.0,
        18182000000.0,
        19161000000.0,
        18028000000.0,
        2344000000.0,
        17571000000.0,
        18123000000.0,
        17560000000.0,
        1926000000.0,
        13187000000.0,
        14218000000.0,
        13251000000.0,
        16694000000.0,
        14197000000.0,
        15535000000.0,
        14107000000.0,
        16690000000.0,
        14252000000.0,
        15475000000.0,
        14752000000.0,
        17381000000.0
    ];
    let input_data_array2 = Array2::from_shape_vec((input.len(), 1), input).unwrap();

    let future_forcast_length = 16;

    let forcasted_time_series = simulate_time_series(input_data_array2, future_forcast_length);
    let forcasted_time_series_vec = forcasted_time_series.into_raw_vec();

    let output_json = serde_json::to_string(&forcasted_time_series_vec).unwrap();
    println!("{}", output_json);
}
