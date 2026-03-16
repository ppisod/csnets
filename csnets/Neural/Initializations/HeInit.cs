using csnets.Neural;
namespace csnets.Neural.Initializations;

/// <summary>
/// He initialization. Weights scaled by sqrt(2 / inputSize).
/// Recommended for ReLU activations.
/// </summary>
public class HeInit : IInitialization {
    public Weight[] InitWeights ( Random random, int inputSize ) {
        Weight[] weights = new Weight[inputSize];
        float scale = (float) Math.Sqrt ( 2.0 / inputSize );
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = new Weight
            {
                value = (float) ( random.NextDouble () * 2 - 1 ) * scale,
                gradients = []
            };
        }
        return weights;
    }
}
