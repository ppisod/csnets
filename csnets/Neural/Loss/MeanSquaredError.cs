namespace csnets.Neural.Loss;

public class MeanSquaredError : ILoss {
    public static float Calculate ( float[] outputs, float[] targets ) {
        float loss = 0;
        for (int i = 0; i < outputs.Length; i++)
        {
            float diff = outputs[i] - targets[i];
            loss += diff * diff;
        }
        return loss * 0.5f;
    }
    public static float[] CalculateDerivative ( float[] outputs, float[] targets ) {
        float[] grad = new float[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
        {
            grad[i] = outputs[i] - targets[i];
        }
        return grad;
    }
}
