namespace csnets.Neural.Loss;

public interface ILoss {
    static abstract float Calculate ( float output, float target );
    static abstract float CalculateDerivative ( float output, float target );
}