use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyArray2};


#[pymodule]
fn functions_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "meshgrid")]
    fn py_meshgrid<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
        let (xv, yv) = rs::meshgrid(x.as_array(), y.as_array());
        Ok((xv.into_pyarray(py), yv.into_pyarray(py)))
    }


    #[pyfn(m)]
    #[pyo3(name = "gaussian7grid")]
    fn py_gaussian7grid<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<f64>,
        xv: PyReadonlyArray2<f64>,
        yv: PyReadonlyArray2<f64>
    ) -> PyResult<&'py PyArray2<f64>> {
        Ok(rs::gaussian7grid(p.as_array(), xv.as_array(), yv.as_array()).into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "gaussian")]
    fn py_gaussian<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<f64>,
        x: usize,
        y: usize
    ) -> PyResult<&'py PyArray2<f64>> {
        Ok(rs::gaussian(p.as_array(), x, y).into_pyarray(py))
    }

    Ok(())
}


mod rs {
    use statrs::function::erf::erf;
    use ndarray::{azip, Array1, Array2, ArrayView1, ArrayView2};


    pub fn meshgrid(x: ArrayView1<f64>, y: ArrayView1<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut xm = Array2::<f64>::zeros((x.len(), y.len()));
        let mut ym = Array2::<f64>::zeros((x.len(), y.len()));
        for i in 0..x.len() {
            for j in 0..y.len() {
                xm[[i, j]] = x[i];
                ym[[i, j]] = y[j];
            }
        }
        (xm, ym)
    }


    fn erf2(x: Array2<f64>) -> Array2<f64> { x.mapv(|i| erf(i)) }


    pub fn gaussian7grid(p: ArrayView1<f64>, xv: ArrayView2<f64>, yv: ArrayView2<f64>) -> Array2<f64> {
        let efac = if p[2] == 0. {
            1e-9
        } else {
            2f64.ln().sqrt() / p[2]
        };
        let (dx, dy) = if p[5] == 0. {
            (efac, efac)
        } else {
            (efac / p[5], efac * p[5])
        };
        let (cos, sin) = (p[6].cos(), p[6].sin());
        let x = 2. * dx * (cos * (&xv - p[0]) - (&yv - p[1]) * sin);
        let y = 2. * dy * (cos * (&yv - p[1]) + (&xv - p[0]) * sin);
        p[3] / 4. * (erf2(&x + dx) - erf2(&x - dx)) * (erf2(&y + dy) - erf2(&y - dy)) + p[4]
    }


    pub fn gaussian(p: ArrayView1<f64>, x: usize, y: usize) -> Array2<f64> {
        let (xv, yv) = meshgrid(
            Array1::<f64>::range(0., y as f64, 1.).view(),
            Array1::<f64>::range(0., x as f64, 1.).view()
        );
        gaussian7grid(p, xv.view(), yv.view())
    }


    // pub fn cost(
    //     p: ArrayView1<f64>,
    //     xv: ArrayView2<f64>,
    //     yv: ArrayView2<f64>,
    //     jm: ArrayView2<f64>
    // ) -> f64 {
    //     let mut out = Array2::<f64>::zeros(jm.shape());
    //     let a = azip!(a in out, b in jm, c in gaussian7grid(p, xv, yv), *a = b - c);
    //
    //     // let a = (jm - gaussian7grid(p, xv, yv).view()).powi(2);
    //
    //     // 1 - np.nansum((jm-gaussian7grid(p, xv, yv))**2) / np.nansum((jm-np.nanmean(jm))**2)
    //     0.
    // }
}