using csnets.Neural.Activations;

namespace csnets.Neural;

public interface INeuron {

    Weight[] weights { get; set; }
    Weight bias { get; set; }

    bool batching { get; set; }

    abstract float ForwardPass <A> (
        float[] inputs,
        bool activate
    ) where A : IActivation;

    abstract BackPropResult BackProp <A> (
        float[] inputs,
        float blame,
        float learningRate,
        bool isLastLayer = false,
        bool updateWeights = true
    ) where A : IActivation;

    abstract void ApplyGradients ( float learningRate );

}

public class Weight {
    public float value;
    public List<float> gradients = [];

    public void AddGradient ( float gradient ) {
        gradients.Add ( gradient );
    }

    public float AverageGradient () {
        if (gradients.Count == 0) return 0;
        float sum = 0;
        foreach (var g in gradients) sum += g;
        return sum / gradients.Count;
    }

    public void ApplyGradients ( float learningRate ) {
        value -= AverageGradient () * learningRate;
        gradients.Clear ();
    }
}

public struct BackPropResult {
    public float blame;
    public float[] nextLayerBlame;
    public float[] weightGradient;
    public float biasGradient;
}
