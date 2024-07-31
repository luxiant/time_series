use ndarray::Array2;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use statrs::{distribution::{ContinuousCDF, FisherSnedecor}, statistics::Statistics};
use rand::prelude::{SeedableRng, StdRng};
use rand_distr::{Distribution, Normal};
use serde_json;
use serde::Deserialize;
use std::io;
use std::io::Read;

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
        let result = (sum_diff_sq / sum_res_sq) as f64;

        result
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
            let a_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = x.dot(&x.t());
            let ax_t_matrix = ax_matrix.t();
            let a_matrix_dmatrix = DMatrix::from_iterator(a_matrix.nrows(), a_matrix.ncols(), a_matrix.iter().map(|x| *x));
            let ax_t_dmatrix_dmatrix = DMatrix::from_iterator(ax_t_matrix.nrows(), ax_t_matrix.ncols(), ax_t_matrix.iter().map(|x| *x));
            let a_matrix_dmatrix_pseudo_inverse = a_matrix_dmatrix.pseudo_inverse(1e-6).unwrap();
            let axr_dmatrix = a_matrix_dmatrix_pseudo_inverse * ax_t_dmatrix_dmatrix.transpose();
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
            (-1.0, dw_stat(res.clone()))
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
    use crate::arima;
    
    pub fn find_adequate_diff(data: Array2<f64>, start_diff: i32) -> Option<i32> {
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let mut diff = start_diff;
        let y = nalgebra::DVector::from_row_slice(&data_slice);
        let regression = Regression::Constant;
        let report = tools::adf_test(&y, diff as usize, regression);
        match report {
            Ok(report) => {
                let critical_value: f64 = distrib::dickeyfuller::get_critical_value(regression, report.size, AlphaLevel::FivePercent).unwrap();
                let t_stat = report.test_statistic;
                if t_stat.abs() >= critical_value.abs() {
                    diff += 1;
                    find_adequate_diff(data, diff)
                } else {
                    Some(diff)
                }
            },
            Err(_) => None,
        }
    }

    pub fn find_adequate_p(data: Array2<f64>) -> (i32, Vec<f64>) {
        let len_data = data.nrows();
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let pacfs = arima::pacf(data_slice, Some(len_data)).unwrap();
        let mut max_p: i32 = 0;
        let mut max_pacf: f64 = pacfs[0];
        for i in 0..pacfs.len() {
            if pacfs[i].abs() > max_pacf.abs() {
                max_pacf = pacfs[i];
                max_p = (i+1) as i32;
            }
        }
        
        (max_p, pacfs)
    }

    pub fn find_adequate_q(data: Array2<f64>) -> (i32, Vec<f64>) {
        let len_data = data.nrows();
        let data_vec = data.clone().into_raw_vec();
        let data_slice = data_vec.as_slice();
        let acfs = arima::acf(data_slice, Some(len_data), true).unwrap();
        let mut max_q: i32 = 0;
        for i in 0..acfs.len() {
            if acfs[i] <= 0.0 {
                break;
            } else {
                max_q = (i+1) as i32;
            }
        }

        (max_q, acfs)
    }
}

pub mod arima {
    use anyhow::Result;
    use std::cmp;
    use rand_distr::{Normal, Distribution};
    use rand::thread_rng;
    use ndarray::Array1;

    pub fn acf(data: &[f64], max_lag: Option<usize>, covariance: bool) -> Result<Vec<f64>> {
        let max_lag = match max_lag {
            Some(max_lag) => cmp::min(max_lag, data.len() - 1),
            None => data.len() - 1,
        };
        let acf_total_len = max_lag + 1;

        let mut sum = 0.0;
        for i in 0..data.len() {
            sum += data[i]
        }
        let data_mean = sum / (data.len() as f64);

        let mut y = vec![0.0; acf_total_len];
        for m in 0..acf_total_len {
            for i in 0..(data.len()-m) {
                let i_th_deviation = data[i] - data_mean;
                let i_t_th_deviation = data[i+m] - data_mean;
                y[m] += (i_th_deviation * i_t_th_deviation) / (data.len() as f64);
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
        let pacf_total_len = max_lag + 1;
        let max_lag_clone = max_lag.clone();
        let rho = acf(data, Some(max_lag_clone), true).unwrap();
        let cov0 = acf(data, Some(0), true).unwrap()[0];

        let mut result_coef: Vec<f64> = Vec::new();

        for i in 1..pacf_total_len {
            let order = cmp::min(i, rho.len() - 1);
            let mut phi: Vec<Vec<f64>> = vec![Vec::new(); order+1];
            let mut var: Vec<f64> = Vec::new();

            phi[0].push(0.0);
            let cov0_clone = cov0.clone();
            var.push(cov0_clone);

            for j in 1..order+1 {
                for _ in 0..j {
                    phi[j].push(0.0);
                }

                let mut num_sum = 0.0;
                let mut den_sum = 1.0;

                for k in 1..j {
                    let p = phi[j-1][k-1];
                    num_sum += p*rho[j-k];
                    den_sum += -p*rho[k];
                }

                let phi_jj = (rho[j]-num_sum)/den_sum;
                phi[j][j-1] = phi_jj;

                var.push(var[j-1]*(1.0 - phi_jj*phi_jj));

                for k in 1..j {
                    phi[j][k-1] = phi[j-1][k-1] - phi[j][j-1]*phi[j-1][j-k-1];
                }
            }

            let coef = phi[order].clone();
            
            result_coef.push(coef[i-1]);
        }

        Ok(result_coef)
    }

    pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
        let mut diff = vec![0.0; data.len()-d];
        for i in d..data.len() {
            diff[i-d] = data[i] - data[i-d];
        }

        diff
    }

    pub fn fit(data: &[f64], p: usize, d: usize, q: usize, _init_params: Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
        let differenced_data = if d == 0 as usize {
            data.to_vec()
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
        let hat_len = differenced_data.len()-max_lag;
        let mut residuals_hat = vec![1.0; hat_len];
        let mut data_hat = vec![0.0; hat_len];

        let mut data_sum = 0.0;
        for i in 0..differenced_data.len() {
            data_sum += differenced_data[i];
        }
        let intercept = data_sum/(differenced_data.len() as f64);

        let mut ar_coef = vec![1.0; p];
        let mut ma_coef = vec![1.0; q];

        // durbin recursion method
        let max_iter = 1000;
        let tol = 1e-6;

        for i in 0..max_iter {
            let residuals_mean = residuals_hat.iter().sum::<f64>()/(residuals_hat.len() as f64);
            for t in 0..q {
                residuals_hat[t] = residuals_mean;
            }
            for t in q..hat_len {
                let mut x_t = 0.0;
                for ar_i in 0..ar_coef.len() {
                    x_t += ar_coef[ar_i]*differenced_data[t+max_lag-ar_i-1];
                }
                for ma_i in 0..ma_coef.len() {
                    x_t += ma_coef[ma_i]*residuals_hat[t-ma_i];
                }
                x_t += intercept;
                data_hat[t] = x_t;
                residuals_hat[t] = differenced_data[t]-x_t;
            }

            let mut new_ar_coef = vec![0.0; ar_coef.len()];
            let mut new_ma_coef = vec![0.0; ma_coef.len()];
            for i in 0..ar_coef.len() {
                let numerator_phi: f64 = (i+1..hat_len).map(|t| differenced_data[t+max_lag] * differenced_data[t+max_lag-i-1]).sum();
                let denominator_phi: f64 = (i+1..hat_len).map(|t| differenced_data[t+max_lag-i-1] * differenced_data[t+max_lag-i-1]).sum();
                new_ar_coef[i] = numerator_phi / denominator_phi;
            }
            for i in 0..ma_coef.len() {
                let numerator_theta: f64 = (i+1..hat_len).map(|t| residuals_hat[t] * residuals_hat[t-i-1]).sum();
                let denominator_theta: f64 = (i+1..hat_len).map(|t| residuals_hat[t-i-1] * residuals_hat[t-i-1]).sum();
                new_ma_coef[i] = numerator_theta / denominator_theta;
            }

            let ar_coef_diff: f64 = new_ar_coef.iter().zip(&ar_coef).map(|(e, p)| (e - p).abs()).sum();
            let ma_coef_diff: f64 = new_ma_coef.iter().zip(&ma_coef).map(|(e, p)| (e - p).abs()).sum();
            if ar_coef_diff.abs() < tol && ma_coef_diff.abs() < tol {
                fitted = true;
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

        let mut hat_differentiated = vec![0.0; differenced_data.len()];
        for i in 0..max_lag {
            hat_differentiated[i] = differenced_data[i];
        }
        for i in max_lag..differenced_data.len() {
            hat_differentiated[i] = data_hat[i-max_lag];
        }

        let mut hat_undifferentiated = vec![0.0; hat_differentiated.len()+d];
        for i in 0..d {
            hat_undifferentiated[i] = data[i];
        }
        for i in 0..data_hat.len() {
            hat_undifferentiated[i+d] = hat_undifferentiated[i] + hat_differentiated[i];
        }

        (result_coef, hat_undifferentiated, residuals_hat, fitted)
    }

    pub fn forecast(simulated_data: &[f64], residuals_hat: &[f64], coef: Vec<f64>, p: usize, d: usize, q: usize, simulation_length: usize) -> Vec<f64> {
        let mut differenced_data = if d == 0 as usize {
            simulated_data.to_vec()
        } else {
            difference(simulated_data, d).as_slice().to_vec()
        };
        let residuals_hat_clone = residuals_hat.to_vec();
        let residual_vec_for_stat = Array1::from(residuals_hat_clone);
        let residual_mean = residual_vec_for_stat.mean().unwrap();
        let residual_std = residual_vec_for_stat.std(0.0);

        let mut residuals_hat_clone = residuals_hat.to_vec();
        let intercept = coef[0];
        let ar_coef = coef[1..1+p].to_vec();
        let ma_coef = coef[1+p..1+p+q].to_vec();

        for _ in 0..simulation_length {
            let normal = Normal::new(residual_mean, residual_std).unwrap();
            let mut rng = thread_rng();
            let random_number = normal.sample(&mut rng);
            residuals_hat_clone.push(random_number);
        }

        for t in 0..simulation_length {
            let mut x_t = 0.0;
            for ar_i in 0..ar_coef.len() {
                x_t += ar_coef[ar_i]*differenced_data[differenced_data.len()+t-ar_i-1];
            }
            for ma_i in 0..ma_coef.len() {
                x_t += ma_coef[ma_i]*residuals_hat_clone[residuals_hat.len()+t-ma_i-1];
            }
            x_t += intercept;
            differenced_data.push(x_t);
        }

        let mut hat_undifferentiated = vec![0.0; differenced_data.len()+d];
        for i in 0..d {
            hat_undifferentiated[i] = differenced_data[i];
        }
        for i in 0..differenced_data.len() {
            hat_undifferentiated[i+d] = hat_undifferentiated[i] + differenced_data[i];
        }

        let result = hat_undifferentiated[hat_undifferentiated.len()-simulation_length..].to_vec();

        result
    }
}

pub mod garch {
    use anyhow::Error;
    use rand::Rng;
    // use finitediff::FiniteDiff;
    use argmin::core::{CostFunction, State, Executor};
    use argmin::solver::neldermead::NelderMead;
    // use argmin::solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS};
    // use argmin::solver::particleswarm::ParticleSwarm;

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

    pub fn fit(ts: &Vec<f64>, p: usize, q: usize, init_params: Vec<f64>) -> Option<Vec<f64>> {
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

        // impl Gradient for StructForGarchFit {
        //     type Param = Vec<f64>;
        //     type Gradient = Vec<f64>;

        //     fn gradient(&self, coef: &Self::Param) -> Result<Self::Gradient, Error> {
        //         let f = |coef: &Vec<f64>| {
        //             let omega = coef[0];
        //             let alpha = &coef[1..self.p+1];
        //             let beta = &coef[self.p+1..];
        //             let sigma_2 = garch_recursion(omega, &alpha, &beta, &self.eps);
        //             neg_loglikelihood(&sigma_2, &self.eps)
        //         };

        //         Ok(coef.forward_diff(&f))
        //     }
        // }

        let p_clone = p.clone();
        let q_clone = q.clone();
        let cost = StructForGarchFit {
            p: p_clone,
            q: q_clone,
            eps: eps,
        };

        // Nah, this does not work.
        // let lower_border = vec![-1.0; 1+p+q];
        // let upper_border = vec![1.0; 1+p+q];
        // let solver = ParticleSwarm::new((lower_border, upper_border), 10000);
    
        // let linesearch = MoreThuenteLineSearch::new().with_bounds(1e-8, f64::INFINITY).unwrap();
        // let solver = LBFGS::new(linesearch, 7);

        // May natural to think initializing nelder mead optimization from acf, pacf coefficient.
        let mut initial_guesses = vec![init_params.clone()];
        for i in 0..init_params.len() {
            let mut new_guess = init_params.clone();
            new_guess[i] = new_guess[i] + 0.01;
            initial_guesses.push(new_guess);
        }
        let solver = NelderMead::new(initial_guesses).with_sd_tolerance(1e-6).unwrap();

        let result = Executor::new(cost, solver)
        .configure(
            |state| state.max_iters(300)
            // |state| state.max_iters(300).param(init_params)
        )
        .run();
        match result {
            Ok(result) => {
                let coef_option = result.state().get_best_param().unwrap().into();
                match coef_option {
                    Some(x) => {
                        let x_clone = x.clone();
                        // let result_coef = x_clone.position;
                        // Some(result_coef)
                        Some(x_clone)
                    },
                    None => None,
                }
            },
            Err(_) => None,
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

fn fit_residuals_arima(residuals_array: Array2<f64>, p: i32, d: i32, q: i32, init_params: Vec<f64>) -> Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let residuals_array_vec = residuals_array.clone().into_raw_vec();
    let residuals_slice = residuals_array_vec.as_slice();
    let (coef, hat_undifferentiated, residuals_hat, fitted) = arima::fit(residuals_slice, p as usize, d as usize, q as usize, init_params);
    
    if fitted {
        let num_params = coef.len() as f64;
        let aic = akaike_information_criterion(num_params, residuals_array.clone());
        
        Some((aic, coef, hat_undifferentiated, residuals_hat))
    } else {
        None
    }
}

fn fit_residuals_garch(residuals_array: Array2<f64>, p: i32, q: i32, init_params: Vec<f64>) -> Option<(f64, Vec<f64>, Array2<f64>)>{
    let residuals_array_vec = residuals_array.clone().into_raw_vec();
    let residuals_slice = residuals_array_vec.clone();
    let coef_option = garch::fit(&residuals_slice, p as usize, q as usize, init_params);
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

fn fit_residuals(rediduals_array: Array2<f64>, is_it_autocorrelated: bool, is_it_heteroskedastic: bool) -> Option<(String, Vec<f64>, Vec<i32>, Array2<f64>)> {
    let residuals_array = rediduals_array.clone();
    let mut result_fitted_residuals: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
    let mut _fitted = false;
    let mut model_type: String;
    let mut coef: Vec<f64> = vec![];
    let mut p_d_q = vec![];
    match (is_it_autocorrelated, is_it_heteroskedastic) {
        (true, false) => {
            model_type = "arima".to_string();
            let residuals_array_clone = residuals_array.clone();
            let (adequate_p, pacfs) = find_p_d_q::find_adequate_p(residuals_array_clone);
            let residuals_array_clone = residuals_array.clone();
            let adequate_diff = find_p_d_q::find_adequate_diff(residuals_array_clone, 0);
            match adequate_diff {
                Some(adequate_diff) => {
                    let residuals_array_clone = residuals_array.clone();
                    let (adequate_q, acfs) = find_p_d_q::find_adequate_q(residuals_array_clone);
            
                    let mut init_params = vec![0.0];
                    for i in 0..(adequate_p as usize) {
                        init_params.push(pacfs[i]);
                    }
                    for i in 0..(adequate_q as usize) {
                        init_params.push(acfs[i]);
                    }
            
                    let mut best_q = 0;
                    let best_aic = f64::INFINITY;
                    let mut check_all_q_are_none = vec![true; (adequate_q+1) as usize];
                    for i in 0..((adequate_q+1) as usize) {
                        let residuals_array_clone = residuals_array.clone();
                        let init_params_clone = init_params.clone();
                        match fit_residuals_arima(residuals_array_clone, adequate_p, adequate_diff, i as i32, init_params_clone) {
                            Some((aic, get_coef, resid_hat_undifferentiated, _resid_residuals_hat)) => {
                                check_all_q_are_none[i] = false;
                                if aic < best_aic {
                                    best_q = i;
                                    result_fitted_residuals = Array2::from_shape_vec((residuals_array.clone().len(), 1), resid_hat_undifferentiated).unwrap();
                                    coef = get_coef;
                                } else {
                                    continue;
                                }
                            },
                            None => continue
                        }
                    }
            
                    _fitted = if check_all_q_are_none == vec![true; (adequate_q+1) as usize] {
                        false
                    } else {
                        true
                    };
                    p_d_q.push(adequate_p);
                    p_d_q.push(adequate_diff);
                    p_d_q.push(best_q as i32);
                },
                None => {
                    _fitted = false;
                    model_type = "none".to_string();
                }
            }
        },
        (false, true) => {
            model_type = "garch".to_string();
            let residuals_array_clone = residuals_array.clone();
            let (adequate_p, pacfs) = find_p_d_q::find_adequate_p(residuals_array_clone);
            let residuals_array_clone = residuals_array.clone();
            let (adequate_q, acfs) = find_p_d_q::find_adequate_q(residuals_array_clone);
    
            let mut init_params = vec![0.0];
            for i in 0..(adequate_p as usize) {
                init_params.push(pacfs[i]);
            }
            for i in 0..(adequate_q as usize) {
                init_params.push(acfs[i]);
            }
    
            let mut best_q = 0;
            let best_aic = f64::INFINITY;
            let mut check_all_q_are_none = vec![true; (adequate_q+1) as usize];
            for i in 0..((adequate_q+1) as usize) {
                let residuals_array_clone = residuals_array.clone();
                let init_params_clone = init_params.clone();
                match fit_residuals_garch(residuals_array_clone, adequate_p, i as i32, init_params_clone) {
                    Some((aic, get_coef, resid_hat)) => {
                        check_all_q_are_none[i] = false;
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
    
            _fitted = if check_all_q_are_none == vec![true; (adequate_q+1) as usize] {
                false
            } else {
                true
            };
            p_d_q.push(adequate_p);
            p_d_q.push(0);
            p_d_q.push(best_q as i32);
        },
        (true, true) => {
            let residuals_array_clone = residuals_array.clone();
            let adequate_diff = find_p_d_q::find_adequate_diff(residuals_array_clone, 0);
            let residuals_array_clone = residuals_array.clone();
            let (adequate_p, pacfs) = find_p_d_q::find_adequate_p(residuals_array_clone);
            let residuals_array_clone = residuals_array.clone();
            let (adequate_q, acfs) = find_p_d_q::find_adequate_q(residuals_array_clone);

            let mut init_params = vec![0.0];
            for i in 0..(adequate_p as usize) {
                init_params.push(pacfs[i]);
            }
            for i in 0..(adequate_q as usize) {
                init_params.push(acfs[i]);
            }

            let mut adequate_diff_usize = 0;
            
            let mut best_q_arima = 0;
            let mut best_fit_residuals_arima: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
            let mut best_arima_coef: Vec<f64> = vec![];
            let mut best_aic_arima = f64::INFINITY;
            let mut _arima_fitted = false;
            match adequate_diff {
                Some(adequate_diff) => {     
                    adequate_diff_usize = adequate_diff.clone() as usize;      
                    let mut check_all_arima_q_are_none = vec![true; (adequate_q+1) as usize];
                    for i in 0..((adequate_q+1) as usize) {
                        let residuals_array_clone = residuals_array.clone();
                        let init_params_clone = init_params.clone();
                        match fit_residuals_arima(residuals_array_clone, adequate_p, adequate_diff, i as i32, init_params_clone) {
                            Some((aic, arima_coef, resid_hat_undifferentiated, _resid_residuals_hat)) => {
                                check_all_arima_q_are_none[i] = false;
                                if aic < best_aic_arima {
                                    best_aic_arima = aic;
                                    best_q_arima = i;
                                    best_arima_coef = arima_coef;
                                    best_fit_residuals_arima = Array2::from_shape_vec((residuals_array.clone().len(), 1), resid_hat_undifferentiated).unwrap();
                                } else {
                                    continue;
                                }
                            },
                            None => continue
                        }
                    }
                    _arima_fitted = if check_all_arima_q_are_none == vec![true; (adequate_q+1) as usize] {
                        false
                    } else {
                        true
                    };
                },
                None => {
                    _arima_fitted = false;
                }
            }
    
            let mut best_q_garch = 0;
            let mut best_fit_residuals_garch: Array2<f64> = Array2::zeros((residuals_array.clone().len(), 1));
            let mut best_garch_coef: Vec<f64> = vec![];
            let best_aic_garch = f64::INFINITY;
            let mut check_all_garch_q_are_none = vec![true; (adequate_q+1) as usize];
            for i in 0..((adequate_q+1) as usize) {
                let residuals_array_clone = residuals_array.clone();
                let init_params_clone = init_params.clone();
                match fit_residuals_garch(residuals_array_clone, adequate_p, i as i32, init_params_clone) {
                    Some((aic, garch_coef, resid_hat)) => {
                        check_all_garch_q_are_none[i] = false;
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
            let garch_fitted = if check_all_garch_q_are_none == vec![true; (adequate_q+1) as usize] {
                false
            } else {
                true
            };
    
            match (_arima_fitted, garch_fitted) {
                (true, true) => {
                    _fitted = true;
                    if best_aic_arima < best_aic_garch {
                        result_fitted_residuals = best_fit_residuals_arima;
                        model_type = "arima".to_string();
                        coef = best_arima_coef;
                        p_d_q.push(adequate_p);
                        p_d_q.push(adequate_diff_usize as i32);
                        p_d_q.push(best_q_arima as i32);
                    } else {
                        result_fitted_residuals = best_fit_residuals_garch;
                        model_type = "garch".to_string();
                        coef = best_garch_coef;
                        p_d_q.push(adequate_p);
                        p_d_q.push(0);
                        p_d_q.push(best_q_garch as i32);
                    }
                },
                (true, false) => {
                    _fitted = true;
                    result_fitted_residuals = best_fit_residuals_arima;
                    model_type = "arima".to_string();
                    coef = best_arima_coef;
                    p_d_q.push(adequate_p);
                    p_d_q.push(adequate_diff_usize as i32);
                    p_d_q.push(best_q_arima as i32);
                },
                (false, true) => {
                    _fitted = true;
                    result_fitted_residuals = best_fit_residuals_garch;
                    model_type = "garch".to_string();
                    coef = best_garch_coef;
                    p_d_q.push(adequate_p);
                    p_d_q.push(0);
                    p_d_q.push(best_q_garch as i32);
                },
                _ => {
                    _fitted = false;
                    result_fitted_residuals = best_fit_residuals_garch;
                    model_type = "garch".to_string();
                    coef = best_garch_coef;
                    p_d_q.push(adequate_p);
                    p_d_q.push(0);
                    p_d_q.push(best_q_garch as i32);
                }
            }
        },
        (false, false) => {
            model_type = "none".to_string();
            _fitted = false;
        },
    }

    match _fitted {
        true => {
            Some((model_type, coef, p_d_q, result_fitted_residuals))
        },
        false => None
    }
}

fn simulate_time_series(input_sample: Array2<f64>, future_forcast_length: i32, is_this_should_be_positive: bool) -> Array2<f64> {
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
    let result_simulated_time_series = time_series_hat_array2.clone();
    let time_residuals = time_model.residuals().to_vec();
    let time_residuals_clone = time_residuals.clone();
    let time_residuals_array2 = Array2::from_shape_vec((time_residuals_clone.len(), 1), time_residuals_clone).unwrap();
    let time_residuals_clone = time_residuals_array2.clone();
    let (is_it_autocorrelated, is_it_heteroskedastic) = check_autocorrelation_and_heteroskedascity_of_residuals(time_model, time_residuals_clone);
    let time_residuals_clone = time_residuals_array2.clone();
    match fit_residuals(time_residuals_clone, is_it_autocorrelated, is_it_heteroskedastic) {
        Some((resid_model_type, resid_coef, resid_p_d_q, fitted_residuals)) => {
            let time_residuals_clone = time_residuals.clone();
            match resid_model_type.as_str() {
                "arima" => {
                    let resid_forecast = arima::forecast(
                        &time_residuals_clone.as_slice(),
                        &fitted_residuals.into_raw_vec().as_slice(),
                        resid_coef,
                        resid_p_d_q[0] as usize,
                        resid_p_d_q[1] as usize,
                        resid_p_d_q[2] as usize,
                        future_forcast_length as usize,
                    );
                    let resid_forecast_array2 = Array2::from_shape_vec((resid_forecast.len(), 1), resid_forecast).unwrap();
                    let tmp_simulated_time_series = result_simulated_time_series.clone() + resid_forecast_array2;

                    if !is_this_should_be_positive {
                        tmp_simulated_time_series
                    } else {
                        let mut tmp_is_positive = true;
                        for i in 0..12 {
                            if tmp_simulated_time_series[[i as usize, 0]] <= 0.0 {
                                tmp_is_positive = false;
                                break;
                            }
                        }
                        if tmp_is_positive {
                            tmp_simulated_time_series
                        } else {
                            result_simulated_time_series
                        }
                    }
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
                    let tmp_simulated_time_series = result_simulated_time_series.clone() + resid_forecast_array2;

                    if !is_this_should_be_positive {
                        tmp_simulated_time_series
                    } else {
                        let mut tmp_is_positive = true;
                        for i in 0..12 {
                            if tmp_simulated_time_series[[i as usize, 0]] <= 0.0 {
                                tmp_is_positive = false;
                                break;
                            }
                        }
                        if tmp_is_positive {
                            tmp_simulated_time_series
                        } else {
                            result_simulated_time_series
                        }
                    }
                }
                _ => {
                    result_simulated_time_series
                }
            }
        },
        None => {
            result_simulated_time_series
        }
    }
}

fn main() {
    #[derive(Deserialize)]
    struct InputData {
        input: Vec<f64>,
        future_forcast_length: i32
    }
    let mut input_json = String::new();
    io::stdin().read_to_string(&mut input_json).expect("Failed to read input JSON");
    let input_data: InputData = serde_json::from_str(&input_json).expect("Failed to parse input JSON");
    let input = input_data.input;
    let input_data_array2 = Array2::from_shape_vec((input.len(), 1), input).unwrap();
    let future_forcast_length = input_data.future_forcast_length;

    // This is for the test. Comment out this when compiling the final product.
    // let input = vec![
    //     1434000000.0,
    //     1446000000.0,
    //     1460000000.0,
    //     1509000000.0,
    //     1475000000.0,
    //     1464000000.0,
    //     1578000000.0,
    //     1587000000.0,
    //     1569000000.0,
    //     1546000000.0,
    //     1555000000.0,
    //     1601000000.0,
    //     1587000000.0,
    //     1534000000.0,
    //     1094000000.0,
    //     1644000000.0,
    //     1548000000.0,
    //     1356000000.0,
    //     1452000000.0,
    //     1402000000.0,
    //     1361000000.0,
    //     1354000000.0,
    //     1320000000.0,
    //     1298000000.0,
    //     1300000000.0,
    //     1287000000.0,
    //     1362000000.0,
    //     1458000000.0,
    //     1465000000.0,
    //     1397000000.0,
    //     1406000000.0,
    //     1484000000.0,
    //     1436000000.0,
    //     1291000000.0,
    //     1378000000.0,
    //     1405000000.0,
    //     1364000000.0,
    //     1252000000.0,
    //     1358000000.0,
    //     1433000000.0,
    //     1407000000.0,
    //     1553000000.0,
    //     1517000000.0,
    //     1625000000.0,
    //     1582000000.0,
    //     1515000000.0,
    //     1540000000.0,
    //     1616000000.0,
    //     1641000000.0,
    //     1606000000.0,
    //     1625000000.0,
    //     1679000000.0,
    //     1673000000.0,
    //     1611000000.0,
    //     1604000000.0,
    //     1655000000.0,
    //     1687000000.0,
    //     1685000000.0,
    //     1748000000.0
    // ];
    // let input_data_array2 = Array2::from_shape_vec((input.len(), 1), input).unwrap();
    // let future_forcast_length = 16;

    let forcasted_time_series = simulate_time_series(input_data_array2, future_forcast_length, true);
    let forcasted_time_series_vec = forcasted_time_series.into_raw_vec();

    let output_json = serde_json::to_string(&forcasted_time_series_vec).unwrap();
    println!("{}", output_json);
}
