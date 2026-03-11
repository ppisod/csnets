namespace csnets.Neural.Loss;

public class MeanSquaredError : ILoss {
    public static float Calculate ( float output, float target ) {
        return ( output - target ) * ( output - target );
    }
    public static float CalculateDerivative ( float output, float target ) {
        return output - target;
    }
}