using csnets.Neural.Activations;
namespace csnets.Neural;

public class DenseLayer {

    public List <Neuron> neurons;

    /// <summary>
    /// initializes random neurons for this dense layer with a given Random
    /// </summary>
    /// <param name="numberOfNeurons">the number of neurons to initialize</param>
    /// <param name="numberOfInputs">the number of inputs to each neuron / the number of neurons in the DenseLayer before this one</param>
    /// <param name="random">the random to use</param>
    public DenseLayer ( int numberOfNeurons, int numberOfInputs, Random random ) {
        neurons = [];
        for (int i = 0; i < numberOfNeurons; i++)
        {
            neurons.Add (new Neuron (random, numberOfInputs));
        }
    }

    public DenseLayer ( int numberOfNeurons, int numberOfInputs ) : this ( numberOfNeurons, numberOfInputs, new Random () ) { }


    public float[] ForwardPass <A> ( float[] inputs, bool isLastLayer ) where A : IActivation {
        float[] outputs = [];
        for (var index = 0; index < neurons.Count; index++)
        {
            var neuron = neurons[index];
            outputs[index] = ( neuron.ForwardPass <A> ( inputs, !isLastLayer ) );
        }

        return outputs;
    }

    public float[] BackProp <A> ( float[] inputs, float[] blame, float learningRate, bool isLastLayer = false ) where A : IActivation {
        float[] lastLayerBlame = new float[inputs.Length];

        for (var index = 0; index < neurons.Count; index++)
        {
            var neuronBlame = neurons[index].BackProp <A> ( inputs, blame[index], learningRate, isLastLayer );

            for (var i = 0; i < lastLayerBlame.Length; i++)
            {
                lastLayerBlame[i] += neuronBlame[i];
            }
        }

        return lastLayerBlame;
    }


}