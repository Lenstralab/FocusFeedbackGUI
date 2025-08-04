use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "functions_rs")]
fn functions_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[allow(clippy::type_complexity)]
    #[pyfn(m)]
    fn meshgrid<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (xv, yv) = rs::meshgrid(x.as_array(), y.as_array());
        Ok((xv.into_pyarray(py), yv.into_pyarray(py)))
    }

    #[pyfn(m)]
    fn gaussian7grid<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<f64>,
        xv: PyReadonlyArray2<f64>,
        yv: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(rs::gaussian7grid(p.as_array(), xv.as_array(), yv.as_array()).into_pyarray(py))
    }

    #[pyfn(m)]
    fn gaussian<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<f64>,
        x: usize,
        y: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(rs::gaussian(p.as_array(), x, y).into_pyarray(py))
    }

    Ok(())
}

mod rs {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use statrs::function::erf::erf;

    pub fn meshgrid(x: ArrayView1<f64>, y: ArrayView1<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut xm = Array2::<f64>::zeros((y.len(), x.len()));
        let mut ym = Array2::<f64>::zeros((y.len(), x.len()));
        for i in 0..y.len() {
            for j in 0..x.len() {
                xm[[i, j]] = x[j];
                ym[[i, j]] = y[i];
            }
        }
        (xm, ym)
    }

    fn erf2(x: Array2<f64>) -> Array2<f64> {
        x.mapv(erf)
    }

    pub fn gaussian7grid(
        p: ArrayView1<f64>,
        xv: ArrayView2<f64>,
        yv: ArrayView2<f64>,
    ) -> Array2<f64> {
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
            Array1::<f64>::range(0., x as f64, 1.).view(),
        );
        gaussian7grid(p, xv.view(), yv.view())
    }
}
