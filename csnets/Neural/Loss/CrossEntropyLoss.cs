namespace csnets.Neural.Loss;

public class CrossEntropyLoss : ILoss {
    public static float Calculate ( float output, float target ) {
        float clamped = Math.Clamp ( output, 1e-7f, 1 - 1e-7f );
        return -( target * MathF.Log ( clamped ) + ( 1 - target ) * MathF.Log ( 1 - clamped ) );
    }
    public static float CalculateDerivative ( float output, float target ) {
        float clamped = Math.Clamp ( output, 1e-7f, 1 - 1e-7f );
        return -( target / clamped ) + ( 1 - target ) / ( 1 - clamped );
    }
}
