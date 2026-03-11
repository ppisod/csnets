using csnets.Neural.Activations;

namespace csnets.Neural;

public interface INeuron {

    float[] weights { get; set; }
    float bias { get; set; }

    abstract float ForwardPass <A> (
        float[] inputs,
        bool activate
    ) where A : IActivation;

    abstract float[] BackProp <A> (
        float[] inputs,
        float blame,
        float learningRate,
        bool isLastLayer = false
    ) where A : IActivation;

}