namespace csnets.Neural.Initializations;

/// <summary>
/// Naive uniform initialization. Weights in [-1, 1].
/// </summary>
public class RandomInit : IInitialization {
    public float[] InitWeights ( Random random, int inputSize ) {
        float[] weights = new float[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = (float) ( random.NextDouble () * 2 - 1 );
        }
        return weights;
    }
}
