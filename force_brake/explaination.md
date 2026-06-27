### Numerical Stability Handling When Normal Force Approaches Zero

When the normal force approaches zero, the resultant tangential force also tends to zero. This can lead to ill-conditioned numerical behavior during scale computation (e.g., `max_f_tan / f_tan`). In subsequent gradient calculations, this may result in gradient explosion and oscillations during training.

To address these issues, several constraints are introduced:

1. **Clamping the normal force with epsilon**
   Apply `max(n_f, eps)` to prevent the maximum tangential force from collapsing to zero.

2. **Epsilon stabilization in denominator**
   Add a small constant `eps` when computing the resultant tangential force to avoid division by near-zero values, which would otherwise cause unstable scaling.

3. **Hard constraint on scaling behavior**
   Enforce a hard constraint between the resultant tangential force and the maximum tangential force, either scaling the tangential force appropriately or keeping it unchanged when stability conditions are violated.
