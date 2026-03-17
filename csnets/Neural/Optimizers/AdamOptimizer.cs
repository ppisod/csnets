namespace csnets.Neural.Optimizers;

public class AdamOptimizer (float meanDecay = 0.9f, float varianceDecay = 0.999f) : IOptimizer {

    public int step = 0;
    public Dictionary <Weight, float> mean = new ();
    public Dictionary <Weight, float> variance = new ();

    public readonly float meanDecay = meanDecay;
    public readonly float varianceDecay = varianceDecay;

    public void ApplyGradient ( Weight weight, float gradient, float learningRate ) {
        step += 1;
        float selectedMean = mean.GetValueOrDefault ( weight, 0 );
        float selectedVariance = variance.GetValueOrDefault ( weight, 0 );
        mean[weight] = meanDecay * selectedMean + (1 - meanDecay) * gradient;
        variance[weight] = varianceDecay * selectedVariance + (1 - varianceDecay) * (gradient * gradient);
        float meanHat = mean[weight] / ( 1 - (float) Math.Pow ( meanDecay, step ) );
        float varHat = variance[weight] / ( 1 - (float) Math.Pow ( varianceDecay, step ) );
        weight.value -= learningRate * meanHat / (float) Math.Sqrt ( varHat + 1e-8f );
    }
    public void ApplyGradients ( Weight weight, float learningRate ) {
        float gradientSum = 0f;
        foreach (var gradient in weight.gradients)
        {
            gradientSum += gradient;
        }
        float gradientMean = gradientSum / weight.gradients.Count;
        ApplyGradient ( weight, gradientMean, learningRate );
        weight.gradients.Clear ();
    }
}