namespace csnets.Neural.Loss;

public interface ILoss {
    static abstract float Calculate ( float[] outputs, float[] targets );
    static abstract float[] CalculateDerivative ( float[] outputs, float[] targets );
}
