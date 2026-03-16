namespace csnets.Neural.Optimizers;

public class SGDOptimizer : IOptimizer {
    public void UpdateWeight ( Weight weight, float gradient, float learningRate ) {
        weight.value -= learningRate * gradient;
    }
    public void ApplyGradients ( Weight weight, float learningRate ) {
        weight.value -= learningRate * weight.AverageGradient ();
        weight.gradients.Clear ();
    }
}