namespace csnets.Neural.Optimizers;

public interface IOptimizer {
    void UpdateWeight ( Weight weight, float gradient, float learningRate );
    void ApplyGradients ( Weight weight, float learningRate );
}