using csnets.Neural.Activations;

namespace csnets.Neural;

public interface INeuron {

    float[] weights { get; set; }
    float bias { get; set; }

    float[] weightAccumulator { get; set; }
    float biasAccumulator { get; set; }

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

    abstract void AccumulateGradients ( float[] inputs, float realBlame );

    abstract void ApplyGradients ( float learningRate, int batches );

}

public struct BackPropResult {
    public float blame;
    public float[] nextLayerBlame;
    public float[] weightGradient;
    public float biasGradient;
}