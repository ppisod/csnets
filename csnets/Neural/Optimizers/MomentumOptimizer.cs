namespace csnets.Neural.Optimizers;

public class MomentumOptimizer ( float momentum ) : IOptimizer {
    private readonly Dictionary<Weight, float> weightVelocities = new ();

    public virtual void ApplyGradient ( Weight weight, float gradient, float learningRate ) {
        float existingVelocity = weightVelocities.GetValueOrDefault ( weight, 0 );
        existingVelocity = (momentum * existingVelocity) - (gradient * learningRate);
        weight.value += existingVelocity;
        weightVelocities[weight] = existingVelocity;
    }

    public virtual void ApplyGradients ( Weight weight, float learningRate ) {
        float avgGrads = weight.AverageGradient ();
        float existingVelocity = weightVelocities.GetValueOrDefault ( weight, 0 );
        existingVelocity = (momentum * existingVelocity) - (avgGrads * learningRate);
        weight.value += existingVelocity;
        weightVelocities[weight] = existingVelocity;
        weight.gradients.Clear ();
    }
}