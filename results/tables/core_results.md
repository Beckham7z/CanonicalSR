# Core results — MF-bench first-20 (n=400, sigma=0.0, seed train/fresh = 0/7, float64)

**SE = 17/20    fresh-data strict = 18/20    total 17.3s**

| idx | true | recovered | arm | nrmse | SE | fresh_relerr |
|---|---|---|---|---|---|---|
| 1 | `x0*(x3*(-x1 + x2) + 1.0)` | `1*x0+1*x0*x2*x3-1*x0*x1*x3` | narrow_snap_r0 | 4.78e-08 | 1 | 3.11e-08 |
| 2 | `x0*exp(x3*(-x1 + x2))` | `exp(log(x0)+x2*x3-x1*x3)` | expwrap | 4.79e-08 | 1 | 3.09e-08 |
| 3 | `x0*(-x1 + x2)*(-x1 + (Abs(x1 - x2)))/(Abs(x1 - x2))` | `1*x0*x2-2*x0*x1` | narrow_multistart_r2 | 7.25e-08 | 0 | 8.69e-08 |
| 4 | `0.166666666666667*x2*x5/x1 + 0.166666666666667*x3*x4/x0` | `(1/6)*x3*x4/x0+(1/6)*x2*x5/x1` | narrow_snap_r0 | 4.90e-08 | 1 | 3.06e-08 |
| 5 | `0.666666666666667*x1*x2/x0` | `(2/3)*x1*x2/x0` | narrow_snap_r0 | 4.72e-08 | 1 | 2.67e-08 |
| 6 | `x0 + 0.5*x1**2/x2` | `(1/2)*x1**2/x2+1*x0` | narrow_snap_r0 | 6.44e-08 | 1 | 4.81e-08 |
| 7 | `x0/(x1*x2*x3)**0.5` | `1.000000018*(x0)/(sqrt(x1)*sqrt(x2)*sqrt(x3))` | logdiff | 6.49e-08 | 1 | 6.07e-08 |
| 8 | `x0*(x2**2*(0.5*x1 - 0.5) + 1.0)` | `(1/2)*x0*x2**2*x1-(1/2)*x0*x2**2+1*x0` | narrow_multistart_r1 | 1.33e-07 | 1 | 1.63e-07 |
| 9 | `x0*sqrt(x3*x4)/(x1*x2*sqrt(x5))` | `1.000000003*(x0*sqrt(x3)*sqrt(x4))/(x1*x2*sqrt(x5))` | logdiff | 5.74e-08 | 1 | 8.18e-08 |
| 10 | `x4*log(x1/x0) - x5*log(x3/x2)` | `-1*x4*log(x0/x1)+1*x5*log(x2/x3)` | narrow_snap_r0 | 3.55e-08 | 1 | 3.35e-08 |
| 11 | `sqrt(x0/((1.0 - x1*(x0 - 1.0)/(x0*x2*(x1/x2 + 0.02)))*(x0 - 1.0)))` | `0.0124324*x0*log(x1/x2)+0.0899287*x0-0.727585*x0**2/x2-0.00134*x0*x2/x1-1.3801e-05*x0**3+2.01877` | narrow_multistart_r3 | 3.67e-01 | 0 | 9.17e-01 |
| 12 | `x0*x1**0.5/x2` | `1.000000033*(x0*sqrt(x1))/(x2)` | logdiff | 6.49e-08 | 1 | 2.30e-08 |
| 13 | `(x0 - x1)/x0` | `-1*x1/x0+1` | narrow_snap_r0 | 2.18e-08 | 1 | 4.19e-08 |
| 14 | `x0*x1/x2` | `(x0*x1)/(x2)` | logspace | 2.44e-08 | 1 | 1.43e-08 |
| 15 | `0.125*x0*x1*x2**2` | `0.1250000012*(x0*x1*(x2)**2)` | logdiff | 8.42e-08 | 1 | 1.28e-07 |
| 16 | `1.4142135623731*sqrt(x0*(x3 - x4)/(x1*x2*x5))` | `sqrt(2)*(sqrt(x0)*sqrt((x3-x4)))/(sqrt(x1)*sqrt(x2)*sqrt(x5))` | logdiff | 4.66e-08 | 1 | 4.18e-08 |
| 17 | `1.4142135623731*x0*sqrt(x1*x4*(x5 - x6)/(x2*x3))` | `sqrt(2)*(x0*sqrt(x1)*sqrt(x4)*sqrt((x5-x6)))/(sqrt(x2)*sqrt(x3))` | logdiff | 4.72e-08 | 1 | 4.08e-08 |
| 18 | `1.4142135623731*x0*sqrt(x1*x4*(x4 - x5)/(x2*x3*x6*x7))` | `sqrt(2)*(x0*sqrt(x1)*sqrt(x4)*sqrt((x4-x5)))/(sqrt(x2)*sqrt(x3)*sqrt(x6)*sqrt(x7))` | logdiff | 7.31e-08 | 1 | 9.95e-08 |
| 19 | `x0*(x0**2*x1 + 1.0)/(x0**2*x1*(2.0*x2 + 1.0) + 1.0)` | `0.177983*x0*log(x1/x2)+0.0103525*x0**2/x1-0.259495*x1+0.197806*x0-0.0315707*x0*x1+0.0218958*x1**2+0.00141984*x0*x2**2+0.719941` | narrow_multistart_r2 | 1.70e-01 | 0 | 1.70e+00 |
| 20 | `x0*(x1**2*x2 + 1.0)` | `1*x0*x1**2*x2+1*x0` | narrow_snap_r0 | 7.10e-08 | 1 | 7.37e-08 |

> `idx3` SE=0 is an Abs-criterion artifact; fresh relerr = 1.0e-7 (numerically exact).