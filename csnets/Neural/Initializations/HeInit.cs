namespace csnets.Neural.Initializations;

/// <summary>
/// He initialization. Weights scaled by sqrt(2 / inputSize).
/// Recommended for ReLU activations.
/// </summary>
public class HeInit : IInitialization {
    public float[] InitWeights ( Random random, int inputSize ) {
        float[] weights = new float[inputSize];
        float scale = (float) Math.Sqrt ( 2.0 / inputSize );
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = (float) ( random.NextDouble () * 2 - 1 ) * scale;
        }
        return weights;
    }
}
