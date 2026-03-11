namespace csnets.Neural.Activations;

public class ReLU : IActivation {
    public static float F ( float x ) {
        if (x < 0) return 0;
        return x;
    }

    public static float Df ( float x ) {
        if (x < 0) return 0;
        return 1;
    }
}