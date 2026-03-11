namespace csnets.Neural.Activations;

public class Sigmoid : IActivation {
    public static float F ( float x ) {
        if (x < -45.0) return 0.0f; // Prevents overflow for large negative values
        if (x > 45.0) return 1.0f;  // Prevents overflow for large positive values
        return 1.0f / (1.0f + (float) Math.Exp(-x));
    }
    public static float Df ( float x ) {
        return x * (1 - x);
    }
}