using csnets.Neural.Activations;
using csnets.Neural.Initializations;
namespace csnets.Neural;

public class DenseLayer {

    public List <INeuron> neurons;

    /// <summary>
    /// initializes random neurons for this dense layer with a given Random
    /// </summary>
    /// <param name="numberOfNeurons">the number of neurons to initialize</param>
    /// <param name="numberOfInputs">the number of inputs to each neuron / the number of neurons in the DenseLayer before this one</param>
    /// <param name="random">the random to use</param>
    public DenseLayer ( int numberOfNeurons, int numberOfInputs, Random random, float? momentum, IInitialization? init = null ) {
        init ??= new HeInit ();
        neurons = [];
        for (int i = 0; i < numberOfNeurons; i++)
        {
            if (momentum != null)
            {
                neurons.Add ( new MomentalNeuron ( random, numberOfInputs, momentum.Value, init ) );
                continue;
            }
            neurons.Add ( new Neuron ( random, numberOfInputs, init ) );
        }
    }

    public DenseLayer ( int numberOfNeurons, int numberOfInputs, float? momentum, IInitialization? init = null ) : this ( numberOfNeurons, numberOfInputs, new Random (), momentum, init ) { }

    public float[] ForwardPass <A> ( float[] inputs, bool isLastLayer ) where A : IActivation {
        float[] outputs = new float[neurons.Count];
        for (var index = 0; index < neurons.Count; index++)
        {
            var neuron = neurons[index];
            outputs[index] = ( neuron.ForwardPass <A> ( inputs, !isLastLayer ) );
        }

        return outputs;
    }

    public float[] BackProp <A> ( float[] inputs, float[] blame, float learningRate, bool isLastLayer = false, bool batching = false ) where A : IActivation {
        float[] lastLayerBlame = new float[inputs.Length];

        for (var index = 0; index < neurons.Count; index++)
        {
            var neuronBlame = neurons[index].BackProp <A> ( inputs, blame[index], learningRate, isLastLayer, !batching );

            for (var i = 0; i < lastLayerBlame.Length; i++)
            {
                lastLayerBlame[i] += neuronBlame.nextLayerBlame[i];
            }
        }

        return lastLayerBlame;
    }

    public void ApplyGradients ( float learningRate ) {
        foreach (var neuron in neurons)
        {
            neuron.ApplyGradients ( learningRate );
        }
    }


}