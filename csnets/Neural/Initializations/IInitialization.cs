namespace csnets.Neural.Initializations;

public interface IInitialization {
    float[] InitWeights ( Random random, int inputSize );
}
