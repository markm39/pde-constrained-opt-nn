# Problems Used

Quick reference for all PDE problems. All 1D heat problems use domain x in (0,1), t in (0,T) with homogeneous Dirichlet BCs.

---

## 1D Poisson Problems

### poisson-1d-scalar
```
-u''(x) = f,  u(0) = u(1) = 0
u(x) = x(1-x)/2
f = -1  (constant)
```

### poisson-1d-vector
```
-u''(x) = f(x),  u(0) = u(1) = 0
u(x) = sin(pi*x)
f(x) = pi^2 * sin(pi*x)
```

---

## 1D Heat Problems

### heat-1d
```
u(x,t) = sin(pi*x) * sin(pi*t)
u0(x) = 0
f(x,t) = pi * sin(pi*x) * [cos(pi*t) + pi*sin(pi*t)]
```
Force is always positive (sin(pi*x) >= 0 for x in [0,1], temporal part positive).

### heat-1d-oscillating
```
u(x,t) = sin(k*pi*x) * sin(pi*t),  u0(x) = 0
f(x,t) = pi * sin(k*pi*x) * [cos(pi*t) + k^2*pi*sin(pi*t)]
```
Parameter: n_oscillations (k). Force oscillates in x, always same sign as sin(k*pi*x).

### heat-1d-oscillating-cosine
```
u(x,t) = sin(omega*pi*x) * cos(omega*pi*t)
u0(x) = sin(omega*pi*x)
f(x,t) = sin(omega*pi*x) * [omega^2*pi^2*cos(omega*pi*t) - omega*pi*sin(omega*pi*t)]
```
Parameter: n_oscillations (omega). Non-zero IC. Force can be positive or negative.

### heat-1d-cosine
```
u(x,t) = sin(2*pi*x) * cos(2*pi*t)
u0(x) = sin(2*pi*x)
f(x,t) = sin(2*pi*x) * [4*pi^2*cos(2*pi*t) - 2*pi*sin(2*pi*t)]
```
Non-zero IC. Force has spatial sign changes (positive for x<0.5, negative for x>0.5).

### heat-1d-mixed
```
u(x,t) = sin(pi*x) * (3t - 4t^2)
u0(x) = 0
f(x,t) = sin(pi*x) * [(3 - 8t) + pi^2*(3t - 4t^2)]
```
Force transitions from positive (early t) to negative (late t, around t>0.65).

### heat-1d-spatial-mixed
```
u(x,t) = sin(2*pi*x) * sin(pi*t)
u0(x) = 0
f(x,t) = sin(2*pi*x) * [pi*cos(pi*t) + 4*pi^2*sin(pi*t)]
```
**Spatial sign changes**: positive for x in (0, 0.5), negative for x in (0.5, 1).

### heat-1d-multimode
```
u(x,t) = [sin(pi*x) - 0.5*sin(2*pi*x)] * sin(pi*t)
u0(x) = 0
f(x,t) = pi*cos(pi*t)*[sin(pi*x) - 0.5*sin(2*pi*x)]
       + pi^2*sin(pi*t)*[sin(pi*x) - 2*sin(2*pi*x)]
```
Multi-mode spatial structure with multiple zero crossings.

### heat-1d-spatial-mixed-nonzero-ic
```
u(x,t) = sin(2*pi*x) * cos(pi*t)
u0(x) = sin(2*pi*x)
f(x,t) = sin(2*pi*x) * [-pi*sin(pi*t) + 4*pi^2*cos(pi*t)]
```
**Non-zero IC with spatial sign changes**. Force has same spatial pattern as u.

---

## 2D Problems

### poisson-2d
```
-div(kappa(x,y) * grad(u)) = f
kappa(x,y) = 1 + 2x + 3y^2
u(x,y) = sin(pi*x) * sin(pi*y)
f(x,y) = complex expression (see code)
```

### linear-heat-2d (default)
```
u_t - Delta(u) = f
u(x,y,t) = exp(t - t^2) * sin(pi*x) * sin(pi*y)
u0(x,y) = sin(pi*x) * sin(pi*y)
f(x,y,t) = (1 - 2t)*exp(t-t^2)*sin(pi*x)*sin(pi*y)
         + 2*pi^2*exp(t-t^2)*sin(pi*x)*sin(pi*y)
```

### linear-heat-2d (cossinsin)
```
u(x,y,t) = sin(5*pi*x) * sin(5*pi*y) * sin(5*pi*t)
u0(x,y) = 0
f(x,y,t) = sin(5*pi*x)*sin(5*pi*y) * [5*pi*cos(5*pi*t) + 50*pi^2*sin(5*pi*t)]
```

### nonlinear-heat-2d
```
u_t - Delta(u) + u^2 = f
u(x,y,t) = exp(t - t^2) * sin(pi*x) * sin(pi*y)
u0(x,y) = sin(pi*x) * sin(pi*y)
f(x,y,t) = (1 - 2t)*exp(t-t^2)*sin(pi*x)*sin(pi*y)
         + 2*pi^2*exp(t-t^2)*sin(pi*x)*sin(pi*y)
         + exp(2(t-t^2))*sin^2(pi*x)*sin^2(pi*y)
```

---

## Other Problems

### wave-1d
```
u_tt - c^2 * u_xx = f
u(x,t) = sin(pi*x) * cos(pi*c*t)
u0(x) = sin(pi*x),  v0(x) = 0
f(x,t) = 0  (homogeneous)
```

### advection-diffusion-1d
```
u_t + v*u_x - D*u_xx = f
u(x,t) = sin(pi*(x - v*t)) * exp(-pi^2*D*t)
u0(x) = sin(pi*x)
f(x,t) = 0  (homogeneous)
```

---

## Summary: Force Sign Patterns

| Problem | f >= 0 everywhere? | Notes |
|---------|-------------------|-------|
| heat-1d | Yes | Safe for ReLU |
| heat-1d-oscillating | No | Oscillates with sin(k*pi*x) |
| heat-1d-oscillating-cosine | No | Mixed signs |
| heat-1d-cosine | No | Spatial sign changes |
| heat-1d-mixed | No | Temporal sign change (~t=0.65) |
| heat-1d-spatial-mixed | No | Spatial sign changes |
| heat-1d-multimode | No | Complex spatial pattern |
| heat-1d-spatial-mixed-nonzero-ic | No | Spatial sign changes |
