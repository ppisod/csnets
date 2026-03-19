namespace csnets.Neural.Loss;

public class SoftmaxCrossEntropy : ILoss {

    public static float[] Softmax ( float[] logits ) {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float[] exp = new float[logits.Length];
        float sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp ( logits[i] - max );
            sum += exp[i];
        }

        for (int i = 0; i < exp.Length; i++)
            exp[i] /= sum;

        return exp;
    }

    public static float Calculate ( float[] outputs, float[] targets ) {
        float[] probs = Softmax ( outputs );
        float loss = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            loss -= targets[i] * MathF.Log ( MathF.Max ( probs[i], 1e-7f ) );
        }
        return loss / outputs.Length;
    }

    /// <summary>
    /// The combined softmax + cross-entropy derivative simplifies to: (softmax(output) - target) / N
    /// </summary>
    public static float[] CalculateDerivative ( float[] outputs, float[] targets ) {
        float[] probs = Softmax ( outputs );
        float[] grad = new float[outputs.Length];
        float n = outputs.Length;
        for (int i = 0; i < outputs.Length; i++)
        {
            grad[i] = ( probs[i] - targets[i] ) / n;
        }
        return grad;
    }

}
